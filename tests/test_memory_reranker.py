import pytest
from pydantic import ValidationError

from memory_engine.memory.reranker import (
    CrossEncoderMemoryReranker,
    FakeMemoryReranker,
    MemoryRerankerSelection,
    select_candidates_by_id,
)
from memory_engine.memory.retriever import RetrievedMemory
from memory_engine.memory.service import Memory, MemoryStatus
from memory_engine.models import Intent, MemoryType


def _candidate(
    memory_id: str,
    content: str,
    *,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> RetrievedMemory:
    return RetrievedMemory(
        memory=Memory(
            id=memory_id,
            content=content,
            memory_type=MemoryType.GOAL,
            confidence=0.95,
            status=status,
        ),
        score=0.8,
    )


class _FakeCrossEncoderScorer:
    def __init__(self, scores: list[float] | Exception) -> None:
        self.scores = scores
        self.calls: list[tuple[tuple[str, str], ...]] = []

    def predict(
        self,
        pairs: tuple[tuple[str, str], ...],
    ) -> list[float]:
        self.calls.append(tuple(pairs))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


@pytest.mark.asyncio
async def test_fake_reranker_retains_only_selected_candidates() -> None:
    message = "I want to learn Japanese."
    japanese = _candidate("japanese", "Learn Japanese")
    spanish = _candidate("spanish", "Improve Spanish")
    reranker = FakeMemoryReranker({message: ["japanese"]})

    result = await reranker.rerank(
        message,
        Intent.GOAL_PLANNING,
        (japanese, spanish),
        2,
    )

    assert result.memories == (japanese,)
    assert result.succeeded
    assert reranker.calls[0][1] is Intent.GOAL_PLANNING


@pytest.mark.asyncio
async def test_fake_reranker_accepts_empty_selection() -> None:
    reranker = FakeMemoryReranker({"unrelated": []})

    result = await reranker.rerank(
        "unrelated",
        Intent.GENERAL_CONVERSATION,
        (_candidate("goal", "A goal"),),
        5,
    )

    assert result.memories == ()
    assert result.succeeded


@pytest.mark.asyncio
async def test_reranker_preserves_selected_order_and_original_objects() -> None:
    first = _candidate("first", "First authoritative content")
    second = _candidate("second", "Second authoritative content")
    reranker = FakeMemoryReranker({"message": ["second", "first"]})

    result = await reranker.rerank(
        "message",
        Intent.REFLECTION,
        (first, second),
        2,
    )

    assert result.memories == (second, first)
    assert result.memories[0] is second
    assert result.memories[1] is first


def test_selection_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown"):
        select_candidates_by_id(
            MemoryRerankerSelection(
                selected_memory_ids=("not-a-candidate",)
            ),
            (_candidate("known", "Known"),),
            1,
        )


def test_selection_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        MemoryRerankerSelection(
            selected_memory_ids=("memory", "memory")
        )


def test_selection_rejects_output_over_limit() -> None:
    candidates = (
        _candidate("first", "First"),
        _candidate("second", "Second"),
    )
    with pytest.raises(ValueError, match="exceeds"):
        select_candidates_by_id(
            MemoryRerankerSelection(
                selected_memory_ids=("first", "second")
            ),
            candidates,
            1,
        )


@pytest.mark.asyncio
async def test_cross_encoder_reranker_batches_and_ranks_candidates() -> None:
    japanese = _candidate("japanese", "The user wants to learn Japanese.")
    spanish = _candidate("spanish", "The user wants to learn Spanish.")
    scorer = _FakeCrossEncoderScorer([8.0, -2.0])
    reranker = CrossEncoderMemoryReranker(
        scorer=scorer,
        relevance_threshold=0.0,
    )

    result = await reranker.rerank(
        "I want to learn Japanese.",
        Intent.GOAL_PLANNING,
        (japanese, spanish),
        2,
    )

    assert result.succeeded
    assert result.memories == (japanese,)
    assert scorer.calls == [
        (
            ("I want to learn Japanese.", japanese.memory.content),
            ("I want to learn Japanese.", spanish.memory.content),
        )
    ]


@pytest.mark.asyncio
async def test_cross_encoder_reranker_supports_none_and_limit() -> None:
    first = _candidate("first", "First")
    second = _candidate("second", "Second")
    reranker = CrossEncoderMemoryReranker(
        scorer=_FakeCrossEncoderScorer([0.4, 0.8]),
        relevance_threshold=0.5,
    )

    result = await reranker.rerank(
        "message",
        Intent.REFLECTION,
        (first, second),
        1,
    )

    assert result.memories == (second,)


@pytest.mark.asyncio
async def test_cross_encoder_reranker_excludes_inactive_candidates() -> None:
    inactive = _candidate(
        "inactive",
        "Inactive",
        status=MemoryStatus.INVALIDATED,
    )
    active = _candidate("active", "Active")
    scorer = _FakeCrossEncoderScorer([0.9])
    reranker = CrossEncoderMemoryReranker(
        scorer=scorer,
        relevance_threshold=0.0,
    )

    result = await reranker.rerank(
        "message",
        Intent.REFLECTION,
        (inactive, active),
        2,
    )

    assert result.memories == (active,)
    assert scorer.calls == [(("message", "Active"),)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [RuntimeError("unavailable"), [float("nan")], []],
)
async def test_cross_encoder_reranker_fails_closed(scores: object) -> None:
    reranker = CrossEncoderMemoryReranker(
        scorer=_FakeCrossEncoderScorer(scores),  # type: ignore[arg-type]
        relevance_threshold=0.0,
    )

    result = await reranker.rerank(
        "message",
        Intent.REFLECTION,
        (_candidate("active", "Active"),),
        2,
    )

    assert not result.succeeded
    assert result.memories == ()
    assert result.error
