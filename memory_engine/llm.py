from abc import ABC, abstractmethod
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import json
import logging
import re
from time import perf_counter
from typing import Any, AsyncIterator
from weakref import WeakSet

import httpx


logger = logging.getLogger(__name__)
_NETWORK_CLIENTS: WeakSet["LLMClient"] = WeakSet()


async def close_network_llm_clients() -> None:
    """Close pooled provider connections created during this process."""

    await asyncio.gather(
        *(client.aclose() for client in tuple(_NETWORK_CLIENTS)),
        return_exceptions=True,
    )


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    api_attempts: int = 1
    retry_attempts: int = 0


def summarize_usage(records: tuple[LLMUsage, ...]) -> dict[str, int]:
    return {
        "model_calls": len(records),
        "input_tokens": sum(item.input_tokens for item in records),
        "cached_input_tokens": sum(
            item.cached_input_tokens for item in records
        ),
        "output_tokens": sum(item.output_tokens for item in records),
        "reasoning_tokens": sum(item.reasoning_tokens for item in records),
        "total_tokens": sum(item.total_tokens for item in records),
        "api_attempts": sum(item.api_attempts for item in records),
        "retry_attempts": sum(item.retry_attempts for item in records),
    }


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    async def stream(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield await self.generate(
            messages,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
        )

    async def aclose(self) -> None:
        return None


class FakeLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        return self.response


class OllamaLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("Maximum retries must not be negative.")
        if retry_backoff_seconds < 0:
            raise ValueError("Retry backoff must not be negative.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.json_mode = json_mode
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.transport = transport
        self._http_client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        )
        _NETWORK_CLIENTS.add(self)
        self._usage_records: list[LLMUsage] = []

    async def generate(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        call_started = perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        if max_output_tokens is not None:
            payload["options"]["num_predict"] = max_output_tokens

        if response_schema is not None:
            payload["format"] = response_schema
        elif self.json_mode:
            payload["format"] = "json"

        try:
            response = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._http_client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    response.raise_for_status()
                    break
                except httpx.TimeoutException:
                    if attempt >= self.max_retries:
                        raise
                except httpx.HTTPStatusError as error:
                    if (
                        attempt >= self.max_retries
                        or not self._retryable_status(
                            error.response.status_code
                        )
                    ):
                        raise
                except httpx.HTTPError:
                    if attempt >= self.max_retries:
                        raise
                await asyncio.sleep(
                    self.retry_backoff_seconds * (2**attempt)
                )
            assert response is not None
            response_data = response.json()

        except httpx.ConnectError as error:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from error

        except httpx.TimeoutException as error:
            raise RuntimeError(
                f"Ollama did not respond within "
                f"{self.timeout_seconds} seconds."
            ) from error

        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                "Ollama returned an HTTP error: "
                f"{error.response.status_code} "
                f"{error.response.text}"
            ) from error

        except ValueError as error:
            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from error

        self._record_usage(response_data, api_attempts=attempt + 1)
        logger.info(
            "llm_call provider=ollama model=%s api_attempts=%d "
            "retry_attempts=%d latency_ms=%.3f",
            self.model,
            attempt + 1,
            attempt,
            (perf_counter() - call_started) * 1000,
        )

        message_data = response_data.get("message")

        if not isinstance(message_data, dict):
            raise RuntimeError(
                "Ollama response did not contain a message."
            )

        generated_text = message_data.get("content")

        if not isinstance(generated_text, str):
            raise RuntimeError(
                "Ollama response did not contain message content."
            )

        return self._extract_final_answer(generated_text)

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    def drain_usage(self) -> tuple[LLMUsage, ...]:
        records = tuple(self._usage_records)
        self._usage_records.clear()
        return records

    async def aclose(self) -> None:
        await self._http_client.aclose()

    def _record_usage(
        self, response_data: dict[str, Any], *, api_attempts: int
    ) -> None:
        prompt_tokens = int(response_data.get("prompt_eval_count", 0))
        output_tokens = int(response_data.get("eval_count", 0))
        self._usage_records.append(
            LLMUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                api_attempts=api_attempts,
                retry_attempts=max(0, api_attempts - 1),
            )
        )

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        cleaned = text.strip()

        # Handles:
        # <think>reasoning...</think>final answer
        cleaned = re.sub(
            r"<think\b[^>]*>.*?</think>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # Handles malformed output:
        # reasoning...</think>final answer
        closing_tag_match = re.search(
            r"</think\s*>",
            cleaned,
            flags=re.IGNORECASE,
        )

        if closing_tag_match:
            cleaned = cleaned[
                closing_tag_match.end():
            ].strip()

        return cleaned


class OpenAIResponsesLLMClient(LLMClient):
    """Provider adapter for OpenAI's Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        reasoning_effort: str = "low",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
        verbosity: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank.")
        if not model.strip():
            raise ValueError("OpenAI model must not be blank.")
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("Unsupported reasoning effort.")
        if max_retries < 0:
            raise ValueError("Maximum retries must not be negative.")
        if retry_backoff_seconds < 0:
            raise ValueError("Retry backoff must not be negative.")
        if verbosity not in {None, "low", "medium", "high"}:
            raise ValueError("Unsupported text verbosity.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.transport = transport
        self.verbosity = verbosity
        self._http_client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        )
        _NETWORK_CLIENTS.add(self)
        self._usage_records: list[LLMUsage] = []

    async def generate(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        call_started = perf_counter()
        payload = self._build_payload(
            messages,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
        )

        try:
            response = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._http_client.post(
                        f"{self.base_url}/responses",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    break
                except httpx.TimeoutException:
                    if attempt >= self.max_retries:
                        raise
                except httpx.HTTPStatusError as error:
                    if (
                        attempt >= self.max_retries
                        or not self._retryable_response(error.response)
                    ):
                        raise
                except httpx.HTTPError:
                    if attempt >= self.max_retries:
                        raise
                await asyncio.sleep(
                    self.retry_backoff_seconds * (2**attempt)
                )
            assert response is not None
            response_data = response.json()
        except httpx.TimeoutException as error:
            raise RuntimeError(
                "OpenAI did not respond within "
                f"{self.timeout_seconds} seconds."
            ) from error
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                "OpenAI returned an HTTP error: "
                f"{error.response.status_code} {error.response.text}"
            ) from error
        except httpx.HTTPError as error:
            raise RuntimeError("Could not connect to OpenAI.") from error
        except ValueError as error:
            raise RuntimeError("OpenAI returned invalid JSON.") from error

        self._record_usage(response_data, api_attempts=attempt + 1)
        logger.info(
            "llm_call provider=openai model=%s api_attempts=%d "
            "retry_attempts=%d latency_ms=%.3f",
            self.model,
            attempt + 1,
            attempt,
            (perf_counter() - call_started) * 1000,
        )

        output_texts: list[str] = []
        refusals: list[str] = []
        for item in response_data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and isinstance(
                    content.get("text"), str
                ):
                    output_texts.append(content["text"])
                elif content.get("type") == "refusal" and isinstance(
                    content.get("refusal"), str
                ):
                    refusals.append(content["refusal"])
        if output_texts:
            return "".join(output_texts).strip()
        if refusals:
            raise RuntimeError(
                "OpenAI refused the request: " + " ".join(refusals)
            )
        raise RuntimeError("OpenAI response did not contain text output.")

    async def stream(
        self,
        messages: list[ChatMessage],
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if response_schema is not None:
            async for value in super().stream(
                messages,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
            ):
                yield value
            return

        call_started = perf_counter()
        payload = self._build_payload(
            messages,
            response_schema=None,
            max_output_tokens=max_output_tokens,
        )
        payload["stream"] = True
        emitted = False
        first_delta_ms: float | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._http_client.stream(
                    "POST",
                    f"{self.base_url}/responses",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw_event = line[5:].strip()
                        if not raw_event or raw_event == "[DONE]":
                            continue
                        try:
                            event = json.loads(raw_event)
                        except ValueError as error:
                            raise RuntimeError(
                                "OpenAI streaming returned invalid JSON."
                            ) from error
                        event_type = event.get("type")
                        if event_type == "response.output_text.delta":
                            delta = event.get("delta")
                            if isinstance(delta, str) and delta:
                                emitted = True
                                if first_delta_ms is None:
                                    first_delta_ms = (
                                        perf_counter() - call_started
                                    ) * 1000
                                yield delta
                        elif event_type == "response.completed":
                            completed = event.get("response")
                            if isinstance(completed, dict):
                                self._record_usage(
                                    completed,
                                    api_attempts=attempt + 1,
                                )
                        elif event_type in {
                            "error",
                            "response.failed",
                            "response.incomplete",
                        }:
                            raise RuntimeError(
                                "OpenAI streaming response failed."
                            )
                if not emitted:
                    raise RuntimeError(
                        "OpenAI streaming response contained no text output."
                    )
                logger.info(
                    "llm_stream provider=openai model=%s api_attempts=%d "
                    "retry_attempts=%d first_delta_ms=%.3f latency_ms=%.3f",
                    self.model,
                    attempt + 1,
                    attempt,
                    first_delta_ms or 0.0,
                    (perf_counter() - call_started) * 1000,
                )
                return
            except httpx.HTTPStatusError as error:
                if emitted or attempt >= self.max_retries or not (
                    self._retryable_response(error.response)
                ):
                    raise RuntimeError(
                        "OpenAI returned an HTTP error while streaming: "
                        f"{error.response.status_code} {error.response.text}"
                    ) from error
            except httpx.TimeoutException as error:
                if emitted or attempt >= self.max_retries:
                    raise RuntimeError(
                        "OpenAI streaming timed out."
                    ) from error
            except httpx.HTTPError as error:
                if emitted or attempt >= self.max_retries:
                    raise RuntimeError(
                        "Could not stream from OpenAI."
                    ) from error
            if emitted:
                raise RuntimeError(
                    "OpenAI stream failed after partial output."
                )
            await asyncio.sleep(
                self.retry_backoff_seconds * (2**attempt)
            )

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        response_schema: dict[str, Any] | None,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": (
                        "developer"
                        if message.role is ChatRole.SYSTEM
                        else message.role.value
                    ),
                    "content": message.content,
                }
                for message in messages
            ],
            "reasoning": {"effort": self.reasoning_effort},
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if self.verbosity is not None:
            payload["text"] = {"verbosity": self.verbosity}
        if response_schema is not None:
            strict_schema = self._to_openai_strict_schema(response_schema)
            payload.setdefault("text", {})["format"] = {
                "type": "json_schema",
                "name": "structured_response",
                "strict": True,
                "schema": strict_schema,
            }
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _retryable_response(response: httpx.Response) -> bool:
        if response.status_code >= 500:
            return True
        if response.status_code != 429:
            return False
        try:
            payload = response.json()
        except ValueError:
            return True
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return True
        error_type = str(error.get("type", "")).casefold()
        error_code = str(error.get("code", "")).casefold()
        permanent_billing_errors = {
            "insufficient_quota",
            "credit_balance_exhausted",
            "billing_hard_limit_reached",
        }
        return not (
            error_type in permanent_billing_errors
            or error_code in permanent_billing_errors
        )

    def drain_usage(self) -> tuple[LLMUsage, ...]:
        records = tuple(self._usage_records)
        self._usage_records.clear()
        return records

    async def aclose(self) -> None:
        await self._http_client.aclose()

    def _record_usage(
        self, response_data: dict[str, Any], *, api_attempts: int
    ) -> None:
        usage = response_data.get("usage")
        if not isinstance(usage, dict):
            return
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")
        self._usage_records.append(
            LLMUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                cached_input_tokens=(
                    int(input_details.get("cached_tokens", 0))
                    if isinstance(input_details, dict)
                    else 0
                ),
                output_tokens=int(usage.get("output_tokens", 0)),
                reasoning_tokens=(
                    int(output_details.get("reasoning_tokens", 0))
                    if isinstance(output_details, dict)
                    else 0
                ),
                total_tokens=int(usage.get("total_tokens", 0)),
                api_attempts=api_attempts,
                retry_attempts=max(0, api_attempts - 1),
            )
        )

    @classmethod
    def _to_openai_strict_schema(
        cls,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize Pydantic JSON Schema for OpenAI strict mode."""
        normalized = deepcopy(schema)

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if "$ref" in node:
                    reference = node["$ref"]
                    node.clear()
                    node["$ref"] = reference
                    return
                node.pop("default", None)
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties)
                    node["additionalProperties"] = False
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(normalized)
        return normalized
