from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import httpx


Embedding = tuple[float, ...]


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(
        self,
        texts: Sequence[str],
    ) -> tuple[Embedding, ...]:
        raise NotImplementedError


class FakeEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        vectors_by_text: Mapping[str, Sequence[float]],
    ) -> None:
        self.vectors_by_text = {
            text: tuple(float(value) for value in vector)
            for text, vector in vectors_by_text.items()
        }
        self.calls: list[tuple[str, ...]] = []

    async def embed(
        self,
        texts: Sequence[str],
    ) -> tuple[Embedding, ...]:
        requested = tuple(texts)
        self.calls.append(requested)
        try:
            return tuple(
                self.vectors_by_text[text] for text in requested
            )
        except KeyError as error:
            raise ValueError(
                f"No fake embedding configured for: {error.args[0]}"
            ) from error


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def embed(
        self,
        texts: Sequence[str],
    ) -> tuple[Embedding, ...]:
        requested = tuple(texts)
        if not requested:
            return ()
        if any(not text.strip() for text in requested):
            raise ValueError("Embedding inputs must not be blank.")

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": list(requested),
                        "truncate": False,
                    },
                )
                response.raise_for_status()
                response_data: dict[str, Any] = response.json()
        except httpx.ConnectError as error:
            raise RuntimeError(
                "Could not connect to Ollama for embeddings. "
                "Make sure Ollama is running."
            ) from error
        except httpx.TimeoutException as error:
            raise RuntimeError(
                "Ollama embedding request timed out after "
                f"{self.timeout_seconds} seconds."
            ) from error
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                "Ollama returned an embedding HTTP error: "
                f"{error.response.status_code} {error.response.text}"
            ) from error
        except ValueError as error:
            raise RuntimeError(
                "Ollama returned invalid embedding JSON."
            ) from error

        embeddings = response_data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Ollama response did not contain embeddings."
            )
        if len(embeddings) != len(requested):
            raise RuntimeError(
                "Ollama returned a different number of embeddings "
                "than inputs."
            )

        parsed: list[Embedding] = []
        dimension: int | None = None
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(
                    "Ollama returned an empty or invalid embedding."
                )
            if not all(
                isinstance(value, int | float) for value in vector
            ):
                raise RuntimeError(
                    "Ollama embedding contained a non-numeric value."
                )
            parsed_vector = tuple(float(value) for value in vector)
            dimension = dimension or len(parsed_vector)
            if len(parsed_vector) != dimension:
                raise RuntimeError(
                    "Ollama returned inconsistent embedding dimensions."
                )
            parsed.append(parsed_vector)

        return tuple(parsed)
