import asyncio

import pytest

from memory_engine.memory.service import (
    InMemoryMemoryService,
    Memory,
    MemoryStatus,
)
from memory_engine.models import MemoryCandidate, MemoryType


USER_ID = "user-1"
OTHER_USER_ID = "user-2"


def make_memory(
    *,
    memory_id: str,
    content: str,
    memory_type: MemoryType,
    confidence: float = 0.95,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        status=status,
    )


def make_service(
    *memories: Memory,
    other_user_memories: tuple[Memory, ...] = (),
) -> InMemoryMemoryService:
    return InMemoryMemoryService(
        memories_by_user={
            USER_ID: tuple(memories),
            OTHER_USER_ID: other_user_memories,
        }
    )


def list_active(
    service: InMemoryMemoryService,
    user_id: str = USER_ID,
) -> tuple[Memory, ...]:
    return asyncio.run(
        service.list_active(user_id=user_id)
    )


def search(
    service: InMemoryMemoryService,
    query: str,
    *,
    limit: int = 10,
    user_id: str = USER_ID,
) -> tuple[Memory, ...]:
    return asyncio.run(
        service.search(
            user_id=user_id,
            query=query,
            limit=limit,
        )
    )


def test_list_active_returns_only_active_memories_for_user() -> None:
    active = make_memory(
        memory_id="active-1",
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
    )
    deleted = make_memory(
        memory_id="deleted-1",
        content="The user previously lived in Munich.",
        memory_type=MemoryType.PERSONAL_FACT,
        status=MemoryStatus.DELETED,
    )
    other_user_memory = make_memory(
        memory_id="other-1",
        content="The other user prefers detailed explanations.",
        memory_type=MemoryType.PREFERENCE,
    )
    service = make_service(
        active,
        deleted,
        other_user_memories=(other_user_memory,),
    )

    assert list_active(service) == (active,)
    assert list_active(service, OTHER_USER_ID) == (
        other_user_memory,
    )


def test_save_creates_active_memory_for_correct_user() -> None:
    service = make_service()
    candidate = MemoryCandidate(
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
        confidence=0.96,
    )

    created = asyncio.run(
        service.save(
            user_id=USER_ID,
            candidate=candidate,
        )
    )

    assert created.id
    assert created.content == candidate.content
    assert created.memory_type is MemoryType.GOAL
    assert created.confidence == 0.96
    assert created.status is MemoryStatus.ACTIVE
    assert list_active(service) == (created,)
    assert list_active(service, OTHER_USER_ID) == ()


def test_update_preserves_id_and_memory_type() -> None:
    existing = make_memory(
        memory_id="fact-1",
        content="The user lives in Munich.",
        memory_type=MemoryType.PERSONAL_FACT,
        confidence=0.90,
    )
    service = make_service(existing)

    updated = asyncio.run(
        service.update(
            user_id=USER_ID,
            memory_id="fact-1",
            content="The user lives in Berlin.",
            confidence=0.98,
        )
    )

    assert updated.id == "fact-1"
    assert updated.content == "The user lives in Berlin."
    assert updated.memory_type is MemoryType.PERSONAL_FACT
    assert updated.confidence == 0.98
    assert updated.status is MemoryStatus.ACTIVE
    assert list_active(service) == (updated,)


def test_update_rejects_missing_or_inactive_memory() -> None:
    deleted = make_memory(
        memory_id="fact-1",
        content="The user lived in Munich.",
        memory_type=MemoryType.PERSONAL_FACT,
        status=MemoryStatus.DELETED,
    )
    service = make_service(deleted)

    with pytest.raises(
        ValueError,
        match="Active memory not found",
    ):
        asyncio.run(
            service.update(
                user_id=USER_ID,
                memory_id="fact-1",
                content="The user lives in Berlin.",
                confidence=0.98,
            )
        )

    with pytest.raises(
        ValueError,
        match="Active memory not found",
    ):
        asyncio.run(
            service.update(
                user_id=OTHER_USER_ID,
                memory_id="fact-1",
                content="The user lives in Berlin.",
                confidence=0.98,
            )
        )


def test_delete_marks_memory_deleted_and_excludes_it_from_active() -> None:
    existing = make_memory(
        memory_id="goal-1",
        content=(
            "The user wants to become comfortable speaking in public."
        ),
        memory_type=MemoryType.GOAL,
    )
    service = make_service(existing)

    deleted = asyncio.run(
        service.delete(
            user_id=USER_ID,
            memory_id="goal-1",
        )
    )

    assert deleted.id == existing.id
    assert deleted.content == existing.content
    assert deleted.memory_type is existing.memory_type
    assert deleted.confidence == existing.confidence
    assert deleted.status is MemoryStatus.DELETED
    assert list_active(service) == ()


def test_delete_rejects_missing_or_already_deleted_memory() -> None:
    existing = make_memory(
        memory_id="goal-1",
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(existing)

    asyncio.run(
        service.delete(
            user_id=USER_ID,
            memory_id="goal-1",
        )
    )

    with pytest.raises(
        ValueError,
        match="Active memory not found",
    ):
        asyncio.run(
            service.delete(
                user_id=USER_ID,
                memory_id="goal-1",
            )
        )


def test_search_returns_relevant_memory_and_excludes_unrelated() -> None:
    public_speaking = make_memory(
        memory_id="goal-1",
        content=(
            "The user wants to become comfortable speaking in public."
        ),
        memory_type=MemoryType.GOAL,
    )
    location = make_memory(
        memory_id="fact-1",
        content="The user lives in Berlin.",
        memory_type=MemoryType.PERSONAL_FACT,
    )
    service = make_service(public_speaking, location)

    results = search(
        service,
        "Public speaking is no longer my goal.",
    )

    assert results == (public_speaking,)


def test_search_excludes_deleted_memories() -> None:
    deleted_goal = make_memory(
        memory_id="goal-1",
        content=(
            "The user wants to become comfortable speaking in public."
        ),
        memory_type=MemoryType.GOAL,
        status=MemoryStatus.DELETED,
    )
    service = make_service(deleted_goal)

    results = search(
        service,
        "Public speaking is no longer my goal.",
    )

    assert results == ()


def test_search_respects_user_boundary() -> None:
    other_user_goal = make_memory(
        memory_id="other-goal-1",
        content=(
            "The other user wants to become comfortable speaking in public."
        ),
        memory_type=MemoryType.GOAL,
    )
    service = make_service(
        other_user_memories=(other_user_goal,),
    )

    assert search(
        service,
        "Public speaking is no longer my goal.",
    ) == ()
    assert search(
        service,
        "Public speaking is no longer my goal.",
        user_id=OTHER_USER_ID,
    ) == (other_user_goal,)


def test_search_respects_limit() -> None:
    first = make_memory(
        memory_id="goal-1",
        content="The user has a public speaking goal.",
        memory_type=MemoryType.GOAL,
    )
    second = make_memory(
        memory_id="pattern-1",
        content="The user often avoids public speaking.",
        memory_type=MemoryType.RECURRING_PATTERN,
    )
    service = make_service(first, second)

    results = search(
        service,
        "public speaking",
        limit=1,
    )

    assert len(results) == 1
    assert results[0] == first


def test_search_uses_stable_insertion_order_for_equal_scores() -> None:
    first = make_memory(
        memory_id="goal-1",
        content="The user practices public speaking weekly.",
        memory_type=MemoryType.GOAL,
    )
    second = make_memory(
        memory_id="goal-2",
        content="The user studies public speaking weekly.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(first, second)

    first_run = search(service, "public speaking")
    second_run = search(service, "public speaking")

    assert first_run == (first, second)
    assert second_run == first_run


@pytest.mark.parametrize(
    ("query", "limit"),
    [
        ("", 10),
        ("the and it", 10),
        ("public speaking", 0),
        ("public speaking", -1),
    ],
)
def test_search_returns_empty_for_unusable_query_or_limit(
    query: str,
    limit: int,
) -> None:
    memory = make_memory(
        memory_id="goal-1",
        content="The user has a public speaking goal.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(memory)

    assert search(
        service,
        query,
        limit=limit,
    ) == ()


def test_purge_removes_all_history_not_just_active_projection() -> None:
    """
    purge() is true erasure, distinct from delete(): no event survives,
    not even a tombstone. See docs/architecture/runtime-memory-decisions.md
    — an append-only deletion tombstone alone is not sufficient privacy
    erasure.
    """
    service = InMemoryMemoryService()
    created = asyncio.run(
        service.save(
            USER_ID,
            MemoryCandidate(
                content="The user has a peanut allergy.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.98,
            ),
        )
    )
    asyncio.run(
        service.update(USER_ID, created.id, "Updated content.", 0.9)
    )

    asyncio.run(service.purge(USER_ID, created.id))

    assert asyncio.run(service.list_events(USER_ID, created.id)) == ()
    assert list_active(service) == ()


def test_purge_unknown_memory_raises() -> None:
    service = InMemoryMemoryService()

    with pytest.raises(ValueError, match="Memory not found"):
        asyncio.run(service.purge(USER_ID, "does-not-exist"))


def test_purge_user_removes_everything_for_that_user_only() -> None:
    kept = make_memory(
        memory_id="other-1",
        content="The other user prefers detailed explanations.",
        memory_type=MemoryType.PREFERENCE,
    )
    service = make_service(
        make_memory(
            memory_id="goal-1",
            content="The user wants to learn Japanese.",
            memory_type=MemoryType.GOAL,
        ),
        other_user_memories=(kept,),
    )
    asyncio.run(service.record_episode(USER_ID, "raw episode text"))

    asyncio.run(service.purge_user(USER_ID))

    assert asyncio.run(service.list_events(USER_ID)) == ()
    assert list_active(service, OTHER_USER_ID) == (kept,)
