import asyncio

from memory_engine.memory.service import (
    InMemoryMemoryService,
    MemoryEventKind,
    MemoryStatus,
)
from memory_engine.models import MemoryCandidate, MemoryType


def test_update_appends_history_instead_of_overwriting() -> None:
    service = InMemoryMemoryService()
    episode = asyncio.run(
        service.record_episode("user-1", "I live in Munich.")
    )
    created = asyncio.run(
        service.save(
            "user-1",
            MemoryCandidate(
                content="The user lives in Munich.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.96,
            ),
            source_episode_id=episode.id,
            evidence="I live in Munich",
        )
    )
    replacement_episode = asyncio.run(
        service.record_episode(
            "user-1",
            "I moved to Berlin instead of Munich.",
        )
    )
    updated = asyncio.run(
        service.update(
            "user-1",
            created.id,
            "The user lives in Berlin.",
            0.98,
            source_episode_id=replacement_episode.id,
            evidence="I moved to Berlin instead of Munich",
        )
    )
    events = asyncio.run(
        service.list_events("user-1", memory_id=created.id)
    )

    assert updated.id == created.id
    assert [event.kind for event in events] == [
        MemoryEventKind.CREATED,
        MemoryEventKind.SUPERSEDED,
    ]
    assert events[0].content == "The user lives in Munich."
    assert events[1].content == "The user lives in Berlin."
    assert events[1].supersedes_event_id == events[0].id


def test_invalidation_preserves_fact_and_removes_current_projection() -> None:
    service = InMemoryMemoryService()
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
            "user-1",
            created.id,
            0.99,
            evidence="I gave up opening a bakery",
        )
    )

    assert invalidated.status is MemoryStatus.INVALIDATED
    assert asyncio.run(service.list_active("user-1")) == ()
    events = asyncio.run(service.list_events("user-1", created.id))
    assert len(events) == 2
    assert events[-1].kind is MemoryEventKind.INVALIDATED
    assert events[-1].content == created.content
