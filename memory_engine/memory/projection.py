from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from memory_engine.memory.service import Memory, MemoryEvent, MemoryEventKind, MemoryStatus
from memory_engine.models import MemoryType

"""
Pure, storage-independent memory logic shared by every MemoryService
backend (in-process and persistent). An event list projects to the same
active state and a query ranks the same way regardless of where the
events actually live — only fetching/appending events differs per
backend.
"""


def memory_from_event(
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


def project_current(events: Sequence[MemoryEvent]) -> tuple[Memory, ...]:
    """The latest event per memory_id, in first-seen order, as a Memory."""
    latest_by_memory: dict[str, MemoryEvent] = {}
    order: list[str] = []
    for event in events:
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
        projected.append(memory_from_event(event, status=status))
    return tuple(projected)


def find_previous_event(
    events: Sequence[MemoryEvent],
    memory_id: str,
) -> MemoryEvent:
    """The most recent event for one memory_id. Assumes at least one exists."""
    return next(
        event
        for event in reversed(events)
        if event.memory_id == memory_id
    )


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "for",
    "from", "i", "in", "is", "it", "my", "of", "on", "one", "the",
    "their", "this", "to", "user", "want", "wants",
}

_SIMPLE_SUFFIXES = ("ing", "ed", "es", "s")

_TYPE_KEYWORDS: dict[MemoryType, set[str]] = {
    MemoryType.GOAL: {"goal", "goals", "want", "wants", "aim", "plan"},
    MemoryType.PREFERENCE: {"prefer", "prefers", "like", "likes", "dislike"},
    MemoryType.DECISION: {"decided", "decision", "choose", "chosen"},
    MemoryType.VALUE: {"value", "values", "important", "believe"},
    MemoryType.RELATIONSHIP: {
        "mother", "father", "sister", "brother", "partner", "friend", "boss",
    },
    MemoryType.RECURRING_PATTERN: {
        "always", "usually", "often", "keep", "repeatedly",
    },
    MemoryType.PERSONAL_FACT: set(),
}


def _normalize_term(term: str) -> str:
    for suffix in _SIMPLE_SUFFIXES:
        if term.endswith(suffix) and len(term) > len(suffix) + 3:
            return term[: -len(suffix)]
    return term


def tokenize(text: str) -> Counter[str]:
    raw_terms = re.findall(r"[a-z0-9]+", text.casefold())
    normalized_terms = [
        _normalize_term(term)
        for term in raw_terms
        if term not in _STOP_WORDS
    ]
    return Counter(term for term in normalized_terms if term)


def _memory_type_bonus(
    query_terms: Counter[str],
    memory_type: MemoryType,
) -> float:
    terms = set(query_terms)
    relevant_keywords = _TYPE_KEYWORDS.get(memory_type, set())
    return 1.0 if terms & relevant_keywords else 0.0


def relevance_score(
    *,
    query_terms: Counter[str],
    memory: Memory,
) -> float:
    memory_terms = tokenize(memory.content)
    if not memory_terms:
        return 0.0

    shared_terms = query_terms.keys() & memory_terms.keys()
    shared_term_score = sum(
        min(query_terms[term], memory_terms[term]) for term in shared_terms
    )

    query_phrase = " ".join(query_terms.keys())
    memory_phrase = memory.content.casefold()
    exact_phrase_bonus = (
        3.0 if query_phrase and query_phrase in memory_phrase else 0.0
    )

    type_bonus = _memory_type_bonus(
        query_terms=query_terms,
        memory_type=memory.memory_type,
    )

    return float(shared_term_score) + exact_phrase_bonus + type_bonus


def rank_by_relevance(
    *,
    query: str,
    memories: Sequence[Memory],
    limit: int,
) -> tuple[Memory, ...]:
    if limit <= 0:
        return ()

    query_terms = tokenize(query)
    if not query_terms:
        return ()

    ranked: list[tuple[float, int, Memory]] = []
    for position, memory in enumerate(memories):
        score = relevance_score(query_terms=query_terms, memory=memory)
        if score <= 0:
            continue
        ranked.append((score, position, memory))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(memory for _, _, memory in ranked[:limit])
