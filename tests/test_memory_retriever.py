import pytest

from memory_engine.embeddings import FakeEmbeddingClient
from memory_engine.memory.retriever import (
    SemanticCandidateRetriever,
    _cosine_similarity,
)
from memory_engine.memory.service import Memory, MemoryStatus
from memory_engine.models import MemoryType


def _memory(
    memory_id: str,
    content: str,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=MemoryType.GOAL,
        confidence=0.95,
        status=status,
    )


@pytest.mark.asyncio
async def test_semantic_retriever_ranks_and_filters_memories() -> None:
    query = "How can I remain consistent with exercise?"
    exercise = _memory(
        "exercise",
        "The user repeatedly skips workouts when work becomes busy.",
    )
    language = _memory(
        "language",
        "The user wants to become fluent in German.",
    )
    invalidated = _memory(
        "old-exercise",
        "The user wants to train for a marathon.",
        MemoryStatus.INVALIDATED,
    )
    embeddings = FakeEmbeddingClient(
        {
            query: [1.0, 0.0],
            exercise.content: [0.95, 0.05],
            language.content: [0.0, 1.0],
            invalidated.content: [1.0, 0.0],
        }
    )
    retriever = SemanticCandidateRetriever(
        embeddings,
        technical_floor=0.0,
    )

    results = await retriever.retrieve(
        query,
        (language, invalidated, exercise),
        limit=5,
    )

    assert [result.memory.id for result in results] == ["exercise"]
    assert results[0].score > 0.99
    assert invalidated.content not in {
        text for call in embeddings.calls for text in call
    }


@pytest.mark.asyncio
async def test_candidate_retriever_returns_top_k_in_score_order() -> None:
    query = "query"
    first = _memory("first", "first")
    second = _memory("second", "second")
    third = _memory("third", "third")
    embeddings = FakeEmbeddingClient(
        {
            query: [1.0, 0.0],
            first.content: [1.0, 0.0],
            second.content: [0.8, 0.2],
            third.content: [0.6, 0.4],
        }
    )
    retriever = SemanticCandidateRetriever(embeddings)

    results = await retriever.retrieve(
        query,
        (third, first, second),
        limit=2,
    )

    assert [result.memory.id for result in results] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_candidate_retriever_rejects_pathological_scores() -> None:
    query = "query"
    zero = _memory("zero", "zero")
    negative = _memory("negative", "negative")
    non_finite = _memory("non-finite", "non-finite")
    valid = _memory("valid", "valid")
    embeddings = FakeEmbeddingClient(
        {
            query: [1.0, 0.0],
            zero.content: [0.0, 0.0],
            negative.content: [-1.0, 0.0],
            non_finite.content: [float("inf"), 0.0],
            valid.content: [0.01, 1.0],
        }
    )
    retriever = SemanticCandidateRetriever(embeddings)

    results = await retriever.retrieve(
        query,
        (zero, negative, non_finite, valid),
    )

    assert [result.memory.id for result in results] == ["valid"]


@pytest.mark.asyncio
async def test_semantic_retriever_caches_memory_embeddings() -> None:
    query = "exercise"
    memory = _memory("exercise", "workout pattern")
    embeddings = FakeEmbeddingClient(
        {query: [1, 0], memory.content: [1, 0]}
    )
    retriever = SemanticCandidateRetriever(embeddings)

    await retriever.retrieve(query, (memory,))
    await retriever.retrieve(query, (memory,))

    assert embeddings.calls == [
        (query,),
        (memory.content,),
        (query,),
    ]


@pytest.mark.asyncio
async def test_semantic_retriever_respects_boundaries() -> None:
    embeddings = FakeEmbeddingClient({})
    retriever = SemanticCandidateRetriever(embeddings)

    assert await retriever.retrieve("", (), limit=5) == ()
    assert await retriever.retrieve("query", (), limit=5) == ()
    assert await retriever.retrieve("query", (), limit=0) == ()
    assert embeddings.calls == []


def test_cosine_similarity_requires_matching_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        _cosine_similarity((1.0, 0.0), (1.0,))


def test_cosine_similarity_computes_known_values() -> None:
    assert _cosine_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert _cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0
    assert _cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == -1.0
    assert _cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0
