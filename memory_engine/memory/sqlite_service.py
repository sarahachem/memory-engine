from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from memory_engine.memory.projection import (
    find_previous_event,
    memory_from_event,
    project_current,
    rank_by_relevance,
)
from memory_engine.memory.service import (
    Memory,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventKind,
    MemoryService,
    MemoryStatus,
)
from memory_engine.models import MemoryCandidate, MemoryType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    source_episode_id TEXT,
    evidence TEXT,
    supersedes_event_id TEXT,
    sequence INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_events_user
    ON memory_events (user_id, sequence);
CREATE INDEX IF NOT EXISTS idx_memory_events_memory
    ON memory_events (user_id, memory_id, sequence);
"""


class SQLiteMemoryService(MemoryService):
    """
    Durable MemoryService backed by an append-only SQLite events table —
    the same event-sourced model as InMemoryMemoryService, persisted to
    disk so memory survives a process restart. See
    docs/architecture/runtime-memory-decisions.md: memory previously
    lived only in process memory, which is fine for a prototype but not
    for a released product where a redeploy must not erase every user's
    history.

    Each operation opens a short-lived connection on a worker thread via
    asyncio.to_thread rather than holding one connection across the
    event loop — sqlite3 connections aren't safe to share across
    threads, and per-call connections keep this simple without adding
    an async driver dependency.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _row_to_event(row: tuple) -> MemoryEvent:
        (
            id_,
            memory_id,
            user_id,
            kind,
            content,
            memory_type,
            confidence,
            created_at,
            source_episode_id,
            evidence,
            supersedes_event_id,
            _sequence,
        ) = row
        return MemoryEvent(
            id=id_,
            memory_id=memory_id,
            user_id=user_id,
            kind=MemoryEventKind(kind),
            content=content,
            memory_type=MemoryType(memory_type),
            confidence=confidence,
            created_at=datetime.fromisoformat(created_at),
            source_episode_id=source_episode_id,
            evidence=evidence,
            supersedes_event_id=supersedes_event_id,
        )

    def _load_events_sync(self, user_id: str) -> list[MemoryEvent]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT id, memory_id, user_id, kind, content, memory_type, "
                "confidence, created_at, source_episode_id, evidence, "
                "supersedes_event_id, sequence FROM memory_events "
                "WHERE user_id = ? ORDER BY sequence ASC",
                (user_id,),
            ).fetchall()
            return [self._row_to_event(row) for row in rows]
        finally:
            connection.close()

    def _append_event_sync(
        self,
        *,
        user_id: str,
        memory_id: str,
        kind: MemoryEventKind,
        content: str,
        memory_type: MemoryType,
        confidence: float,
        source_episode_id: str | None,
        evidence: str | None,
        supersedes_event_id: str | None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            id=str(uuid4()),
            memory_id=memory_id,
            user_id=user_id,
            kind=kind,
            content=content,
            memory_type=memory_type,
            confidence=confidence,
            created_at=self._now(),
            source_episode_id=source_episode_id,
            evidence=evidence,
            supersedes_event_id=supersedes_event_id,
        )
        connection = self._connect()
        try:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM memory_events"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO memory_events (id, memory_id, user_id, kind, "
                "content, memory_type, confidence, created_at, "
                "source_episode_id, evidence, supersedes_event_id, "
                "sequence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.memory_id,
                    event.user_id,
                    event.kind.value,
                    event.content,
                    event.memory_type.value,
                    event.confidence,
                    event.created_at.isoformat(),
                    event.source_episode_id,
                    event.evidence,
                    event.supersedes_event_id,
                    sequence,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return event

    def _record_episode_sync(
        self,
        user_id: str,
        content: str,
    ) -> MemoryEpisode:
        episode = MemoryEpisode(
            id=str(uuid4()),
            user_id=user_id,
            content=content,
            created_at=self._now(),
        )
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO memory_episodes (id, user_id, content, "
                "created_at) VALUES (?, ?, ?, ?)",
                (
                    episode.id,
                    episode.user_id,
                    episode.content,
                    episode.created_at.isoformat(),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return episode

    def _purge_memory_sync(self, user_id: str, memory_id: str) -> int:
        connection = self._connect()
        try:
            cursor = connection.execute(
                "DELETE FROM memory_events WHERE user_id = ? "
                "AND memory_id = ?",
                (user_id, memory_id),
            )
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _purge_user_sync(self, user_id: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "DELETE FROM memory_events WHERE user_id = ?",
                (user_id,),
            )
            connection.execute(
                "DELETE FROM memory_episodes WHERE user_id = ?",
                (user_id,),
            )
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------ API

    async def record_episode(
        self,
        user_id: str,
        content: str,
    ) -> MemoryEpisode:
        return await asyncio.to_thread(
            self._record_episode_sync, user_id, content
        )

    async def list_events(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> tuple[MemoryEvent, ...]:
        events = await asyncio.to_thread(self._load_events_sync, user_id)
        if memory_id is None:
            return tuple(events)
        return tuple(
            event for event in events if event.memory_id == memory_id
        )

    async def list_active(self, user_id: str) -> tuple[Memory, ...]:
        events = await asyncio.to_thread(self._load_events_sync, user_id)
        return tuple(
            memory
            for memory in project_current(events)
            if memory.status is MemoryStatus.ACTIVE
        )

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> tuple[Memory, ...]:
        active = await self.list_active(user_id)
        return rank_by_relevance(query=query, memories=active, limit=limit)

    async def save(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        event = await asyncio.to_thread(
            self._append_event_sync,
            user_id=user_id,
            memory_id=str(uuid4()),
            kind=MemoryEventKind.CREATED,
            content=candidate.content,
            memory_type=candidate.memory_type,
            confidence=candidate.confidence,
            source_episode_id=source_episode_id,
            evidence=evidence,
            supersedes_event_id=None,
        )
        return memory_from_event(event)

    async def _require_active(
        self,
        user_id: str,
        memory_id: str,
    ) -> tuple[Memory, MemoryEvent]:
        events = await asyncio.to_thread(self._load_events_sync, user_id)
        current = next(
            (
                memory
                for memory in project_current(events)
                if memory.id == memory_id
                and memory.status is MemoryStatus.ACTIVE
            ),
            None,
        )
        if current is None:
            raise ValueError(f"Active memory not found: {memory_id}")
        previous_event = find_previous_event(events, memory_id)
        return current, previous_event

    async def update(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        confidence: float,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        current, previous_event = await self._require_active(
            user_id, memory_id
        )
        event = await asyncio.to_thread(
            self._append_event_sync,
            user_id=user_id,
            memory_id=memory_id,
            kind=MemoryEventKind.SUPERSEDED,
            content=content,
            memory_type=current.memory_type,
            confidence=confidence,
            source_episode_id=source_episode_id,
            evidence=evidence,
            supersedes_event_id=previous_event.id,
        )
        return memory_from_event(event)

    async def invalidate(
        self,
        user_id: str,
        memory_id: str,
        confidence: float,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        current, previous_event = await self._require_active(
            user_id, memory_id
        )
        event = await asyncio.to_thread(
            self._append_event_sync,
            user_id=user_id,
            memory_id=memory_id,
            kind=MemoryEventKind.INVALIDATED,
            content=current.content,
            memory_type=current.memory_type,
            confidence=confidence,
            source_episode_id=source_episode_id,
            evidence=evidence,
            supersedes_event_id=previous_event.id,
        )
        return memory_from_event(event, status=MemoryStatus.INVALIDATED)

    async def delete(self, user_id: str, memory_id: str) -> Memory:
        current, previous_event = await self._require_active(
            user_id, memory_id
        )
        event = await asyncio.to_thread(
            self._append_event_sync,
            user_id=user_id,
            memory_id=memory_id,
            kind=MemoryEventKind.DELETED,
            content=current.content,
            memory_type=current.memory_type,
            confidence=current.confidence,
            source_episode_id=None,
            evidence=None,
            supersedes_event_id=previous_event.id,
        )
        return memory_from_event(event, status=MemoryStatus.DELETED)

    async def purge(self, user_id: str, memory_id: str) -> None:
        removed = await asyncio.to_thread(
            self._purge_memory_sync, user_id, memory_id
        )
        if removed == 0:
            raise ValueError(f"Memory not found: {memory_id}")

    async def purge_user(self, user_id: str) -> None:
        await asyncio.to_thread(self._purge_user_sync, user_id)
