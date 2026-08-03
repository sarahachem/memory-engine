from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from memory_engine.memory.reranker import MemoryReranker
from memory_engine.memory.retriever import MemoryRetriever, RetrievedMemory
from memory_engine.memory.service import Memory
from memory_engine.models import Intent

logger = logging.getLogger(__name__)


class ContextMemoryRetrievalError(RuntimeError):
    """Infrastructure or contract failure, distinct from valid no-match."""


class FinalContextMemoryRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        *,
        message: str,
        intent: Intent,
        candidate_memories: Sequence[Memory],
        limit: int = 5,
    ) -> tuple[RetrievedMemory, ...]:
        raise NotImplementedError


class ContextMemoryRetriever(FinalContextMemoryRetriever):
    """Compose high-recall candidates with precision-focused reranking."""

    def __init__(
        self,
        candidate_retriever: MemoryRetriever,
        reranker: MemoryReranker,
        candidate_limit: int = 10,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than 0.")
        self.candidate_retriever = candidate_retriever
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    async def retrieve(
        self,
        *,
        message: str,
        intent: Intent,
        candidate_memories: Sequence[Memory],
        limit: int = 5,
    ) -> tuple[RetrievedMemory, ...]:
        if limit <= 0 or not message.strip() or not candidate_memories:
            return ()

        try:
            candidates = await self.candidate_retriever.retrieve(
                query=message,
                candidate_memories=candidate_memories,
                limit=self.candidate_limit,
            )
        except Exception:
            logger.exception(
                "Context candidate retrieval failed closed; no memories "
                "will enter the response context."
            )
            raise ContextMemoryRetrievalError(
                "Context candidate retrieval failed."
            )

        if not candidates:
            return ()

        try:
            result = await self.reranker.rerank(
                message=message,
                intent=intent,
                candidates=candidates,
                limit=limit,
            )
            if not result.succeeded:
                logger.error(
                    "Context memory reranking failed closed: %s",
                    result.error,
                )
                raise ContextMemoryRetrievalError(
                    result.error or "Context reranking failed."
                )
            return result.memories
        except ContextMemoryRetrievalError:
            raise
        except Exception as error:
            logger.exception(
                "Context memory reranking failed closed; no memories "
                "will enter the response context."
            )
            raise ContextMemoryRetrievalError(
                "Context memory reranking failed."
            ) from error
