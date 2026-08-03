import pytest

from memory_engine.embeddings import FakeEmbeddingClient
from memory_engine.memory.context_retriever import (
    ContextMemoryRetrievalError,
    ContextMemoryRetriever,
)
from memory_engine.memory.reranker import (
    FakeMemoryReranker,
    MemoryReranker,
    MemoryRerankerResult,
)
from memory_engine.memory.retriever import SemanticCandidateRetriever
from memory_engine.memory.service import Memory
from memory_engine.models import Intent, MemoryType


def _memory(memory_id: str, content: str) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=MemoryType.GOAL,
        confidence=0.95,
    )


class FailingReranker(MemoryReranker):
    async def rerank(self, message, intent, candidates, limit):
        raise RuntimeError("reranker unavailable")


class RecordingReranker(MemoryReranker):
    def __init__(self) -> None:
        self.calls = []

    async def rerank(self, message, intent, candidates, limit):
        self.calls.append((message, intent, candidates, limit))
        return MemoryRerankerResult(memories=candidates[:limit])


@pytest.mark.asyncio
async def test_context_retriever_composes_candidates_and_reranking() -> None:
    message = "I want to learn Japanese."
    japanese = _memory("japanese", "The user wants to learn Japanese.")
    spanish = _memory("spanish", "The user wants to improve Spanish.")
    candidate_retriever = SemanticCandidateRetriever(
        FakeEmbeddingClient(
            {
                message: [1.0, 0.0],
                japanese.content: [1.0, 0.0],
                spanish.content: [0.9, 0.1],
            }
        )
    )
    final_retriever = ContextMemoryRetriever(
        candidate_retriever,
        FakeMemoryReranker({message: ["japanese"]}),
    )

    result = await final_retriever.retrieve(
        message=message,
        intent=Intent.GOAL_PLANNING,
        candidate_memories=(japanese, spanish),
        limit=1,
    )

    assert [item.memory.id for item in result] == ["japanese"]


@pytest.mark.asyncio
async def test_context_retriever_failure_returns_no_context(caplog) -> None:
    message = "message"
    memory = _memory("memory", "Memory")
    final_retriever = ContextMemoryRetriever(
        SemanticCandidateRetriever(
            FakeEmbeddingClient(
                {message: [1, 0], memory.content: [1, 0]}
            )
        ),
        FailingReranker(),
    )

    with pytest.raises(ContextMemoryRetrievalError):
        await final_retriever.retrieve(
            message=message,
            intent=Intent.REFLECTION,
            candidate_memories=(memory,),
        )
    assert "reranking failed closed" in caplog.text


@pytest.mark.asyncio
async def test_context_retriever_skips_reranker_without_candidates() -> None:
    reranker = RecordingReranker()
    final_retriever = ContextMemoryRetriever(
        SemanticCandidateRetriever(FakeEmbeddingClient({})),
        reranker,
    )

    result = await final_retriever.retrieve(
        message="message",
        intent=Intent.REFLECTION,
        candidate_memories=(),
    )

    assert result == ()
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_context_retriever_candidate_failure_returns_no_context(
    caplog,
) -> None:
    memory = _memory("memory", "Memory")
    reranker = RecordingReranker()
    final_retriever = ContextMemoryRetriever(
        SemanticCandidateRetriever(FakeEmbeddingClient({})),
        reranker,
    )

    with pytest.raises(ContextMemoryRetrievalError):
        await final_retriever.retrieve(
            message="missing embedding",
            intent=Intent.REFLECTION,
            candidate_memories=(memory,),
        )
    assert reranker.calls == []
    assert "candidate retrieval failed closed" in caplog.text
