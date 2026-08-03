from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from memory_engine.models import MemoryCandidate, MemoryType


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    DELETED = "deleted"


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    memory_type: MemoryType
    confidence: float
    status: MemoryStatus = MemoryStatus.ACTIVE
    source_episode_id: str | None = None
    evidence: str | None = None
    # Timestamp of the event that produced this projection — i.e. when this
    # memory was last created/corrected, not necessarily its original
    # creation time. None only for call sites that build a Memory without
    # going through an event (rare; see _memory_from_event).
    # Projection timestamps are event metadata, not part of logical memory
    # identity. Callers may still inspect the value directly.
    created_at: datetime | None = field(default=None, compare=False)


@dataclass(frozen=True)
class MemoryEpisode:
    """
    Immutable record of the user input from which memories were derived.
    """

    id: str
    user_id: str
    content: str
    created_at: datetime


class MemoryEventKind(StrEnum):
    CREATED = "created"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    DELETED = "deleted"


@dataclass(frozen=True)
class MemoryEvent:
    """
    Append-only change event for one logical memory.

    SUPERSEDED contains the replacement value. INVALIDATED preserves the
    previous value but removes it from the current active projection. DELETED
    is reserved for explicit privacy or administrative deletion.
    """

    id: str
    memory_id: str
    user_id: str
    kind: MemoryEventKind
    content: str
    memory_type: MemoryType
    confidence: float
    created_at: datetime
    source_episode_id: str | None = None
    evidence: str | None = None
    supersedes_event_id: str | None = None


class MemoryService(ABC):
    async def record_episode(
        self,
        user_id: str,
        content: str,
    ) -> MemoryEpisode:
        raise NotImplementedError(
            "This memory service does not support episode provenance."
        )

    async def list_events(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> tuple[MemoryEvent, ...]:
        raise NotImplementedError(
            "This memory service does not expose an event history."
        )

    @abstractmethod
    async def list_active(
        self,
        user_id: str,
    ) -> tuple[Memory, ...]:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> tuple[Memory, ...]:
        """
        Return a bounded set of active memories that may be relevant
        to the raw user message.

        This method retrieves memories only. It does not decide whether
        memories should be created, updated, or deleted.
        """
        raise NotImplementedError

    @abstractmethod
    async def save(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    async def invalidate(
        self,
        user_id: str,
        memory_id: str,
        confidence: float,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        # Compatibility fallback for stores that have not migrated to temporal
        # invalidation yet. Production stores should override this method.
        return await self.delete(
            user_id=user_id,
            memory_id=memory_id,
        )

    @abstractmethod
    async def delete(
        self,
        user_id: str,
        memory_id: str,
    ) -> Memory:
        raise NotImplementedError

    async def purge(self, user_id: str, memory_id: str) -> None:
        """
        True erasure: physically removes every stored event for one
        memory rather than appending another tombstone event. Distinct
        from delete(), which preserves history for audit — this exists
        for an explicit right-to-erasure request. An append-only
        deletion tombstone is not sufficient privacy erasure on its own.
        """
        raise NotImplementedError(
            "This memory service does not support true erasure."
        )

    async def purge_user(self, user_id: str) -> None:
        """Erases every stored memory event and episode for one user."""
        raise NotImplementedError(
            "This memory service does not support true erasure."
        )


class InMemoryMemoryService(MemoryService):
    def __init__(
        self,
        memories_by_user: dict[str, tuple[Memory, ...]]
        | None = None,
    ) -> None:
        self._episodes_by_user: dict[str, list[MemoryEpisode]] = {}
        self._events_by_user: dict[str, list[MemoryEvent]] = {}

        for user_id, memories in (memories_by_user or {}).items():
            for memory in memories:
                kind = (
                    MemoryEventKind.CREATED
                    if memory.status == MemoryStatus.ACTIVE
                    else MemoryEventKind.DELETED
                )
                self._append_event(
                    user_id=user_id,
                    memory_id=memory.id,
                    kind=kind,
                    content=memory.content,
                    memory_type=memory.memory_type,
                    confidence=memory.confidence,
                    source_episode_id=memory.source_episode_id,
                    evidence=memory.evidence,
                )

    async def record_episode(
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
        self._episodes_by_user.setdefault(user_id, []).append(episode)
        return episode

    async def list_events(
        self,
        user_id: str,
        memory_id: str | None = None,
    ) -> tuple[MemoryEvent, ...]:
        events = self._events_by_user.get(user_id, [])
        if memory_id is None:
            return tuple(events)
        return tuple(
            event for event in events if event.memory_id == memory_id
        )

    async def list_active(
        self,
        user_id: str,
    ) -> tuple[Memory, ...]:
        return tuple(
            memory
            for memory in self._project_current(user_id)
            if memory.status == MemoryStatus.ACTIVE
        )

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> tuple[Memory, ...]:
        if limit <= 0:
            return ()

        active_memories = await self.list_active(
            user_id=user_id,
        )

        if not active_memories:
            return ()

        query_terms = self._tokenize(query)

        if not query_terms:
            return ()

        ranked_memories: list[
            tuple[float, int, Memory]
        ] = []

        for position, memory in enumerate(
            active_memories
        ):
            score = self._relevance_score(
                query_terms=query_terms,
                memory=memory,
            )

            if score <= 0:
                continue

            ranked_memories.append(
                (
                    score,
                    position,
                    memory,
                )
            )

        ranked_memories.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return tuple(
            memory
            for _, _, memory in ranked_memories[
                :limit
            ]
        )

    async def save(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        memory_id = str(uuid4())
        event = self._append_event(
            user_id=user_id,
            memory_id=memory_id,
            kind=MemoryEventKind.CREATED,
            content=candidate.content,
            memory_type=candidate.memory_type,
            confidence=candidate.confidence,
            source_episode_id=source_episode_id,
            evidence=evidence,
        )
        return self._memory_from_event(event)

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
        current, previous_event = self._require_active(
            user_id=user_id,
            memory_id=memory_id,
        )
        event = self._append_event(
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
        return self._memory_from_event(event)

    async def invalidate(
        self,
        user_id: str,
        memory_id: str,
        confidence: float,
        *,
        source_episode_id: str | None = None,
        evidence: str | None = None,
    ) -> Memory:
        current, previous_event = self._require_active(
            user_id=user_id,
            memory_id=memory_id,
        )
        event = self._append_event(
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
        return self._memory_from_event(
            event,
            status=MemoryStatus.INVALIDATED,
        )

    async def delete(
        self,
        user_id: str,
        memory_id: str,
    ) -> Memory:
        current, previous_event = self._require_active(
            user_id=user_id,
            memory_id=memory_id,
        )
        event = self._append_event(
            user_id=user_id,
            memory_id=memory_id,
            kind=MemoryEventKind.DELETED,
            content=current.content,
            memory_type=current.memory_type,
            confidence=current.confidence,
            supersedes_event_id=previous_event.id,
        )
        return self._memory_from_event(event, status=MemoryStatus.DELETED)

    async def purge(self, user_id: str, memory_id: str) -> None:
        events = self._events_by_user.get(user_id, [])
        remaining = [
            event for event in events if event.memory_id != memory_id
        ]
        if len(remaining) == len(events):
            raise ValueError(f"Memory not found: {memory_id}")
        self._events_by_user[user_id] = remaining

    async def purge_user(self, user_id: str) -> None:
        self._events_by_user.pop(user_id, None)
        self._episodes_by_user.pop(user_id, None)

    def _project_current(self, user_id: str) -> tuple[Memory, ...]:
        latest_by_memory: dict[str, MemoryEvent] = {}
        order: list[str] = []
        for event in self._events_by_user.get(user_id, []):
            if event.memory_id not in latest_by_memory:
                order.append(event.memory_id)
            latest_by_memory[event.memory_id] = event

        projected: list[Memory] = []
        for memory_id in order:
            event = latest_by_memory[memory_id]
            status = MemoryStatus.ACTIVE
            if event.kind == MemoryEventKind.INVALIDATED:
                status = MemoryStatus.INVALIDATED
            elif event.kind == MemoryEventKind.DELETED:
                status = MemoryStatus.DELETED
            projected.append(self._memory_from_event(event, status=status))
        return tuple(projected)

    def _require_active(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> tuple[Memory, MemoryEvent]:
        current = next(
            (
                memory
                for memory in self._project_current(user_id)
                if memory.id == memory_id
                and memory.status == MemoryStatus.ACTIVE
            ),
            None,
        )
        if current is None:
            raise ValueError(f"Active memory not found: {memory_id}")

        previous_event = next(
            event
            for event in reversed(self._events_by_user[user_id])
            if event.memory_id == memory_id
        )
        return current, previous_event

    def _append_event(
        self,
        *,
        user_id: str,
        memory_id: str,
        kind: MemoryEventKind,
        content: str,
        memory_type: MemoryType,
        confidence: float,
        source_episode_id: str | None = None,
        evidence: str | None = None,
        supersedes_event_id: str | None = None,
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
        self._events_by_user.setdefault(user_id, []).append(event)
        return event

    @staticmethod
    def _memory_from_event(
        event: MemoryEvent,
        *,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> Memory:
        return Memory(
            id=event.memory_id,
            content=event.content,
            memory_type=event.memory_type,
            confidence=event.confidence,
            status=status,
            source_episode_id=event.source_episode_id,
            evidence=event.evidence,
            created_at=event.created_at,
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _relevance_score(
        cls,
        query_terms: Counter[str],
        memory: Memory,
    ) -> float:
        memory_terms = cls._tokenize(
            memory.content
        )

        if not memory_terms:
            return 0.0

        shared_terms = (
            query_terms.keys()
            & memory_terms.keys()
        )

        shared_term_score = sum(
            min(
                query_terms[term],
                memory_terms[term],
            )
            for term in shared_terms
        )

        query_phrase = " ".join(
            query_terms.keys()
        )
        memory_phrase = memory.content.casefold()

        exact_phrase_bonus = (
            3.0
            if query_phrase
            and query_phrase in memory_phrase
            else 0.0
        )

        type_bonus = cls._memory_type_bonus(
            query_terms=query_terms,
            memory_type=memory.memory_type,
        )

        return (
            float(shared_term_score)
            + exact_phrase_bonus
            + type_bonus
        )

    @staticmethod
    def _memory_type_bonus(
        query_terms: Counter[str],
        memory_type: MemoryType,
    ) -> float:
        terms = set(query_terms)

        type_keywords: dict[
            MemoryType,
            set[str],
        ] = {
            MemoryType.GOAL: {
                "goal",
                "goals",
                "want",
                "wants",
                "aim",
                "plan",
            },
            MemoryType.PREFERENCE: {
                "prefer",
                "prefers",
                "like",
                "likes",
                "dislike",
            },
            MemoryType.DECISION: {
                "decided",
                "decision",
                "choose",
                "chosen",
            },
            MemoryType.VALUE: {
                "value",
                "values",
                "important",
                "believe",
            },
            MemoryType.RELATIONSHIP: {
                "mother",
                "father",
                "sister",
                "brother",
                "partner",
                "friend",
                "boss",
            },
            MemoryType.RECURRING_PATTERN: {
                "always",
                "usually",
                "often",
                "keep",
                "repeatedly",
            },
            MemoryType.PERSONAL_FACT: set(),
        }

        relevant_keywords = type_keywords.get(
            memory_type,
            set(),
        )

        return (
            1.0
            if terms & relevant_keywords
            else 0.0
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> Counter[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "because",
            "but",
            "for",
            "from",
            "i",
            "in",
            "is",
            "it",
            "my",
            "of",
            "on",
            "one",
            "the",
            "their",
            "this",
            "to",
            "user",
            "want",
            "wants",
        }

        raw_terms = re.findall(
            r"[a-z0-9]+",
            text.casefold(),
        )

        normalized_terms = [
            InMemoryMemoryService._normalize_term(
                term
            )
            for term in raw_terms
            if term not in stop_words
        ]

        return Counter(
            term
            for term in normalized_terms
            if term
        )

    @staticmethod
    def _normalize_term(
        term: str,
    ) -> str:
        simple_suffixes = (
            "ing",
            "ed",
            "es",
            "s",
        )

        for suffix in simple_suffixes:
            if (
                term.endswith(suffix)
                and len(term) > len(suffix) + 3
            ):
                return term[
                    : -len(suffix)
                ]

        return term
