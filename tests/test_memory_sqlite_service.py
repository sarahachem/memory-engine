import asyncio
from pathlib import Path

import pytest

from memory_engine.memory.service import MemoryEventKind, MemoryStatus
from memory_engine.memory.sqlite_service import SQLiteMemoryService
from memory_engine.models import MemoryCandidate, MemoryType


def _service(tmp_path: Path) -> SQLiteMemoryService:
    return SQLiteMemoryService(tmp_path / "memory.sqlite3")


def test_save_creates_active_memory(tmp_path: Path) -> None:
    service = _service(tmp_path)
    candidate = MemoryCandidate(
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
        confidence=0.96,
    )

    created = asyncio.run(service.save("user-1", candidate))

    assert created.id
    assert created.content == candidate.content
    assert created.status is MemoryStatus.ACTIVE
    assert asyncio.run(service.list_active("user-1")) == (created,)


def test_memory_survives_a_new_service_instance_against_same_file(
    tmp_path: Path,
) -> None:
    """The whole point of this backend: a fresh process must see prior data."""
    db_path = tmp_path / "memory.sqlite3"
    first = SQLiteMemoryService(db_path)
    candidate = MemoryCandidate(
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
        confidence=0.96,
    )
    created = asyncio.run(first.save("user-1", candidate))

    second = SQLiteMemoryService(db_path)
    reloaded = asyncio.run(second.list_active("user-1"))

    assert reloaded == (created,)


def test_list_active_respects_user_boundary(tmp_path: Path) -> None:
    service = _service(tmp_path)
    asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user wants to learn Japanese.",
                memory_type=MemoryType.GOAL,
                confidence=0.95,
            ),
        )
    )
    asyncio.run(
        service.save(
            "user-2",
            MemoryCandidate(
                content="The other user prefers detailed explanations.",
                memory_type=MemoryType.PREFERENCE,
                confidence=0.95,
            ),
        )
    )

    assert len(asyncio.run(service.list_active("user-1"))) == 1
    assert len(asyncio.run(service.list_active("user-2"))) == 1
    first = asyncio.run(service.list_active("user-1"))[0]
    second = asyncio.run(service.list_active("user-2"))[0]
    assert first.content != second.content


def test_update_appends_history_instead_of_overwriting(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    created = asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user lives in Munich.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.96,
            ),
            evidence="I live in Munich",
        )
    )

    updated = asyncio.run(
        service.update(
            "user-1",
            created.id,
            "The user lives in Berlin.",
            0.98,
            evidence="I moved to Berlin instead of Munich",
        )
    )
    events = asyncio.run(service.list_events("user-1", memory_id=created.id))

    assert updated.id == created.id
    assert updated.content == "The user lives in Berlin."
    assert [event.kind for event in events] == [
        MemoryEventKind.CREATED,
        MemoryEventKind.SUPERSEDED,
    ]
    assert events[0].content == "The user lives in Munich."
    assert events[1].content == "The user lives in Berlin."
    assert events[1].supersedes_event_id == events[0].id
    assert asyncio.run(service.list_active("user-1")) == (updated,)


def test_update_rejects_missing_memory(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="Active memory not found"):
        asyncio.run(
            service.update("user-1", "does-not-exist", "New content.", 0.9)
        )


def test_invalidate_preserves_fact_and_removes_from_active(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    created = asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user wants to open a bakery.",
                memory_type=MemoryType.GOAL,
                confidence=0.96,
            ),
        )
    )

    invalidated = asyncio.run(
        service.invalidate(
            "user-1", created.id, 0.99, evidence="I gave up on the bakery"
        )
    )
    events = asyncio.run(service.list_events("user-1", created.id))

    assert invalidated.status is MemoryStatus.INVALIDATED
    assert asyncio.run(service.list_active("user-1")) == ()
    assert events[-1].kind is MemoryEventKind.INVALIDATED
    assert events[-1].content == created.content


def test_delete_marks_deleted_and_preserves_history(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user wants to learn Japanese.",
                memory_type=MemoryType.GOAL,
                confidence=0.95,
            ),
        )
    )

    deleted = asyncio.run(service.delete("user-1", created.id))
    events = asyncio.run(service.list_events("user-1", created.id))

    assert deleted.status is MemoryStatus.DELETED
    assert asyncio.run(service.list_active("user-1")) == ()
    # History still exists — delete() is a tombstone, not erasure.
    assert len(events) == 2
    assert events[-1].kind is MemoryEventKind.DELETED


def test_search_ranks_relevant_memory_over_unrelated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content=(
                    "The user wants to become comfortable speaking in "
                    "public."
                ),
                memory_type=MemoryType.GOAL,
                confidence=0.95,
            ),
        )
    )
    asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user lives in Berlin.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.95,
            ),
        )
    )

    results = asyncio.run(
        service.search("user-1", "Public speaking is no longer my goal.")
    )

    assert len(results) == 1
    assert "public" in results[0].content.casefold()


def test_purge_physically_removes_all_history(tmp_path: Path) -> None:
    """
    purge() is true erasure, distinct from delete(): no event survives,
    not even a tombstone. See docs/architecture/runtime-memory-decisions.md
    — an append-only deletion tombstone alone is not sufficient privacy
    erasure.
    """
    service = _service(tmp_path)
    created = asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user has a peanut allergy.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.98,
            ),
        )
    )
    asyncio.run(
        service.update("user-1", created.id, "Updated content.", 0.9)
    )

    asyncio.run(service.purge("user-1", created.id))

    assert asyncio.run(service.list_events("user-1", created.id)) == ()
    assert asyncio.run(service.list_active("user-1")) == ()


def test_purge_unknown_memory_raises(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="Memory not found"):
        asyncio.run(service.purge("user-1", "does-not-exist"))


def test_purge_user_removes_everything_for_that_user_only(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user wants to learn Japanese.",
                memory_type=MemoryType.GOAL,
                confidence=0.95,
            ),
        )
    )
    kept = asyncio.run(
        service.save(
            "user-2",
            MemoryCandidate(
                content="The other user prefers detailed explanations.",
                memory_type=MemoryType.PREFERENCE,
                confidence=0.95,
            ),
        )
    )
    asyncio.run(service.record_episode("user-1", "raw episode text"))

    asyncio.run(service.purge_user("user-1"))

    assert asyncio.run(service.list_events("user-1")) == ()
    assert asyncio.run(service.list_active("user-2")) == (kept,)
