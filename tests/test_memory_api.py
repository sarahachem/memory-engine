import asyncio

from fastapi.testclient import TestClient

from memory_engine.dependencies import get_memory_service
from memory_engine.main import create_app
from memory_engine.memory.service import InMemoryMemoryService
from memory_engine.models import MemoryCandidate, MemoryType


def _client_with_memory(service: InMemoryMemoryService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_memory_service] = lambda: service
    return TestClient(app)


def test_list_memories_returns_provenance() -> None:
    service = InMemoryMemoryService()
    client = _client_with_memory(service)

    memory = asyncio.run(
        service.save(
            user_id="user-1",
            candidate=MemoryCandidate(
                content="Prefers async communication over calls.",
                memory_type=MemoryType.PREFERENCE,
                confidence=0.9,
            ),
            evidence="I really don't like phone calls, email works better.",
        )
    )

    response = client.get("/memory/user-1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == memory.id
    assert body[0]["content"] == "Prefers async communication over calls."
    assert body[0]["memory_type"] == "preference"
    assert body[0]["source_quote"] == (
        "I really don't like phone calls, email works better."
    )
    assert body[0]["created_at"] is not None


def test_list_memories_empty_for_unknown_user() -> None:
    client = _client_with_memory(InMemoryMemoryService())

    response = client.get("/memory/nobody")

    assert response.status_code == 200
    assert response.json() == []


def test_correct_memory_updates_content_and_sets_full_confidence() -> None:
    service = InMemoryMemoryService()
    client = _client_with_memory(service)

    memory = asyncio.run(
        service.save(
            user_id="user-1",
            candidate=MemoryCandidate(
                content="Wrong content the extractor guessed.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.6,
            ),
            evidence="Something the user actually said.",
        )
    )

    response = client.patch(
        f"/memory/user-1/{memory.id}",
        json={"content": "Corrected, user-authored content."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "Corrected, user-authored content."
    assert body["confidence"] == 1.0
    # Provenance survives a correction — what was originally said is
    # still true even after the interpretation of it is fixed.
    assert body["source_quote"] == "Something the user actually said."

    listing = client.get("/memory/user-1").json()
    assert len(listing) == 1
    assert listing[0]["content"] == "Corrected, user-authored content."


def test_correct_unknown_memory_returns_404() -> None:
    client = _client_with_memory(InMemoryMemoryService())

    response = client.patch(
        "/memory/user-1/does-not-exist",
        json={"content": "Anything"},
    )

    assert response.status_code == 404


def test_delete_memory_removes_it_from_listing() -> None:
    service = InMemoryMemoryService()
    client = _client_with_memory(service)

    memory = asyncio.run(
        service.save(
            user_id="user-1",
            candidate=MemoryCandidate(
                content="Something to delete.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.9,
            ),
        )
    )

    response = client.delete(f"/memory/user-1/{memory.id}")

    assert response.status_code == 204
    assert client.get("/memory/user-1").json() == []


def test_delete_memory_is_true_erasure_not_a_tombstone() -> None:
    """
    The user-facing delete endpoint must call purge(), not delete() —
    no event should survive, matching the real-erasure requirement in
    docs/architecture/runtime-memory-decisions.md.
    """
    service = InMemoryMemoryService()
    client = _client_with_memory(service)

    memory = asyncio.run(
        service.save(
            user_id="user-1",
            candidate=MemoryCandidate(
                content="Something to delete.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.9,
            ),
        )
    )

    client.delete(f"/memory/user-1/{memory.id}")

    events = asyncio.run(
        service.list_events("user-1", memory_id=memory.id)
    )
    assert events == ()


def test_delete_unknown_memory_returns_404() -> None:
    client = _client_with_memory(InMemoryMemoryService())

    response = client.delete("/memory/user-1/does-not-exist")

    assert response.status_code == 404


def test_delete_all_memories_erases_only_the_requested_user() -> None:
    service = InMemoryMemoryService()
    client = _client_with_memory(service)

    asyncio.run(
        service.save(
            user_id="user-1",
            candidate=MemoryCandidate(
                content="Something to erase.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.9,
            ),
        )
    )
    kept = asyncio.run(
        service.save(
            user_id="user-2",
            candidate=MemoryCandidate(
                content="Something to keep.",
                memory_type=MemoryType.PERSONAL_FACT,
                confidence=0.9,
            ),
        )
    )

    response = client.delete("/memory/user-1")

    assert response.status_code == 204
    assert client.get("/memory/user-1").json() == []
    assert client.get("/memory/user-2").json()[0]["id"] == kept.id


def test_delete_all_memories_for_user_with_none_is_a_no_op() -> None:
    client = _client_with_memory(InMemoryMemoryService())

    response = client.delete("/memory/user-with-nothing-stored")

    assert response.status_code == 204
