from __future__ import annotations

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.memory.retriever import RetrievedMemory
from memory_engine.memory.service import MemoryStatus
from memory_engine.models import Intent

if TYPE_CHECKING:
    from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryRerankerResult:
    memories: tuple[RetrievedMemory, ...] = ()
    succeeded: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        if self.succeeded and self.error is not None:
            raise ValueError("A successful rerank result cannot have an error.")
        if not self.succeeded and self.error is None:
            raise ValueError("A failed rerank result requires an error.")
        if not self.succeeded and self.memories:
            raise ValueError("A failed rerank result cannot select memories.")


class MemoryRerankerSelection(BaseModel):
    """Minimal structured output shared by future reranker providers."""

    model_config = ConfigDict(extra="forbid")

    selected_memory_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_unique_non_blank_ids(self) -> "MemoryRerankerSelection":
        if any(not memory_id.strip() for memory_id in self.selected_memory_ids):
            raise ValueError("Selected memory IDs must not be blank.")
        if len(self.selected_memory_ids) != len(set(self.selected_memory_ids)):
            raise ValueError("Selected memory IDs must be unique.")
        return self


class MemoryReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        message: str,
        intent: Intent,
        candidates: tuple[RetrievedMemory, ...],
        limit: int,
    ) -> MemoryRerankerResult:
        """Return selected candidates plus an observable success status."""
        raise NotImplementedError


class CrossEncoderScorer(Protocol):
    """Synchronous scoring boundary kept separate from async orchestration."""

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[float]: ...


class SentenceTransformerCrossEncoderScorer:
    """Lazy adapter for the optional sentence-transformers dependency."""

    def __init__(self, model: str, *, local_files_only: bool = True) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - installation boundary
            raise RuntimeError(
                "Cross-encoder reranking requires the optional 'reranker' "
                "dependencies. Install the project with '[reranker]'."
            ) from error
        self._model = CrossEncoder(
            model,
            local_files_only=local_files_only,
        )

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[float]:
        return self._model.predict(list(pairs))


class CrossEncoderMemoryReranker(MemoryReranker):
    """Local batched relevance scorer with a first-class empty selection."""

    def __init__(
        self,
        *,
        scorer: CrossEncoderScorer,
        relevance_threshold: float,
    ) -> None:
        if not math.isfinite(relevance_threshold):
            raise ValueError("Cross-encoder threshold must be finite.")
        self.scorer = scorer
        self.relevance_threshold = relevance_threshold

    async def rerank(
        self,
        message: str,
        intent: Intent,
        candidates: tuple[RetrievedMemory, ...],
        limit: int,
    ) -> MemoryRerankerResult:
        del intent  # Relevance is scored from the query and authoritative text.
        if limit < 0:
            raise ValueError("Reranker limit must not be negative.")
        active_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.memory.status is MemoryStatus.ACTIVE
        )
        if limit == 0 or not active_candidates:
            return MemoryRerankerResult()

        pairs = tuple(
            (message, candidate.memory.content)
            for candidate in active_candidates
        )
        try:
            raw_scores = await asyncio.to_thread(self.scorer.predict, pairs)
            scores = tuple(float(score) for score in raw_scores)
            if len(scores) != len(active_candidates):
                raise ValueError(
                    "Cross-encoder returned a different number of scores "
                    "than candidates."
                )
            if any(not math.isfinite(score) for score in scores):
                raise ValueError("Cross-encoder returned a non-finite score.")

            ranked = sorted(
                (
                    (score, index, candidate)
                    for index, (candidate, score) in enumerate(
                        zip(active_candidates, scores, strict=True)
                    )
                    if score >= self.relevance_threshold
                ),
                key=lambda item: (-item[0], item[1]),
            )
            return MemoryRerankerResult(
                memories=tuple(item[2] for item in ranked[:limit])
            )
        except (RuntimeError, TypeError, ValueError) as error:
            logger.exception(
                "Cross-encoder memory reranker failed closed; no candidate "
                "memories were selected."
            )
            return MemoryRerankerResult(
                succeeded=False,
                error=f"{type(error).__name__}: {error}",
            )


def select_candidates_by_id(
    selection: MemoryRerankerSelection,
    candidates: tuple[RetrievedMemory, ...],
    limit: int,
) -> tuple[RetrievedMemory, ...]:
    if limit < 0:
        raise ValueError("Reranker limit must not be negative.")
    if len(selection.selected_memory_ids) > limit:
        raise ValueError("Reranker selection exceeds the requested limit.")

    candidates_by_id = {
        candidate.memory.id: candidate for candidate in candidates
    }
    unknown_ids = [
        memory_id
        for memory_id in selection.selected_memory_ids
        if memory_id not in candidates_by_id
    ]
    if unknown_ids:
        raise ValueError(
            "Reranker selected unknown memory IDs: "
            + ", ".join(unknown_ids)
        )

    return tuple(
        candidates_by_id[memory_id]
        for memory_id in selection.selected_memory_ids
    )


class FakeMemoryReranker(MemoryReranker):
    """Deterministic reranker for unit tests and evaluator development."""

    def __init__(
        self,
        selected_ids_by_message: Mapping[str, Sequence[str]],
    ) -> None:
        self.selected_ids_by_message = {
            message: tuple(memory_ids)
            for message, memory_ids in selected_ids_by_message.items()
        }
        self.calls: list[
            tuple[str, Intent, tuple[RetrievedMemory, ...], int]
        ] = []

    async def rerank(
        self,
        message: str,
        intent: Intent,
        candidates: tuple[RetrievedMemory, ...],
        limit: int,
    ) -> MemoryRerankerResult:
        self.calls.append((message, intent, candidates, limit))
        if message not in self.selected_ids_by_message:
            raise ValueError(
                f"No fake reranker output configured for: {message}"
            )
        selection = MemoryRerankerSelection(
            selected_memory_ids=self.selected_ids_by_message[message]
        )
        return MemoryRerankerResult(
            memories=select_candidates_by_id(selection, candidates, limit)
        )


class LLMMemoryReranker(MemoryReranker):
    """Semantic, ID-only final relevance judgment that fails closed."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def rerank(
        self,
        message: str,
        intent: Intent,
        candidates: tuple[RetrievedMemory, ...],
        limit: int,
    ) -> MemoryRerankerResult:
        if limit < 0:
            raise ValueError("Reranker limit must not be negative.")
        active_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.memory.status is MemoryStatus.ACTIVE
        )
        if limit == 0 or not active_candidates:
            return MemoryRerankerResult()

        payload = {
            "message": message,
            "intent": intent.value,
            "selection_limit": limit,
            "candidates": [
                {
                    "id": candidate.memory.id,
                    "content": candidate.memory.content,
                    "memory_type": candidate.memory.memory_type.value,
                }
                for candidate in active_candidates
            ],
        }
        try:
            raw_response = await self.llm_client.generate(
                messages=[
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        content=self.prompt_repository.load(
                            "memory/rerank_context"
                        ),
                    ),
                    ChatMessage(
                        role=ChatRole.USER,
                        content=json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ],
                response_schema=(
                    MemoryRerankerSelection.model_json_schema()
                ),
            )
            selection = MemoryRerankerSelection.model_validate_json(
                raw_response
            )
            return MemoryRerankerResult(
                memories=select_candidates_by_id(
                    selection,
                    active_candidates,
                    limit,
                )
            )
        except (RuntimeError, ValidationError, ValueError) as error:
            logger.exception(
                "Memory reranker failed closed; no candidate memories "
                "were selected."
            )
            return MemoryRerankerResult(
                succeeded=False,
                error=f"{type(error).__name__}: {error}",
            )
