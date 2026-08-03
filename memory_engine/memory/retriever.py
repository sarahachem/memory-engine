from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math

from memory_engine.embeddings import Embedding, EmbeddingClient
from memory_engine.memory.service import Memory, MemoryStatus


@dataclass(frozen=True)
class RetrievedMemory:
    memory: Memory
    score: float


class MemoryRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        candidate_memories: Sequence[Memory],
        limit: int = 5,
    ) -> tuple[RetrievedMemory, ...]:
        raise NotImplementedError


class SemanticCandidateRetriever(MemoryRetriever):
    """High-recall semantic candidate generation, not final relevance."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        technical_floor: float = 0.0,
    ) -> None:
        if not 0.0 <= technical_floor < 1.0:
            raise ValueError(
                "technical_floor must be between 0 inclusive and 1 exclusive."
            )
        self.embedding_client = embedding_client
        self.technical_floor = technical_floor
        self._embedding_cache: dict[str, Embedding] = {}

    async def retrieve(
        self,
        query: str,
        candidate_memories: Sequence[Memory],
        limit: int = 5,
    ) -> tuple[RetrievedMemory, ...]:
        if limit <= 0 or not query.strip():
            return ()

        active_memories = tuple(
            memory
            for memory in candidate_memories
            if memory.status is MemoryStatus.ACTIVE
        )
        if not active_memories:
            return ()

        query_embedding = (
            await self.embedding_client.embed((query,))
        )[0]
        memory_embeddings = await self._embed_memories(active_memories)
        ranked: list[RetrievedMemory] = []
        for memory, memory_embedding in zip(
            active_memories,
            memory_embeddings,
            strict=True,
        ):
            score = _cosine_similarity(
                query_embedding,
                memory_embedding,
            )
            if not math.isfinite(score):
                continue
            if score <= self.technical_floor:
                continue
            ranked.append(RetrievedMemory(memory=memory, score=score))
        ranked.sort(key=lambda result: -result.score)
        return tuple(ranked[:limit])

    async def _embed_memories(
        self,
        memories: Sequence[Memory],
    ) -> tuple[Embedding, ...]:
        missing_contents = tuple(
            dict.fromkeys(
                memory.content
                for memory in memories
                if memory.content not in self._embedding_cache
            )
        )
        if missing_contents:
            generated = await self.embedding_client.embed(
                missing_contents
            )
            self._embedding_cache.update(
                zip(missing_contents, generated, strict=True)
            )
        return tuple(
            self._embedding_cache[memory.content]
            for memory in memories
        )


def _cosine_similarity(left: Embedding, right: Embedding) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match.")
    if not left:
        raise ValueError("Embeddings must not be empty.")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
