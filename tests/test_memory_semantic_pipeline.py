import asyncio
import json

import pytest

from memory_engine.llm import FakeLLMClient
from memory_engine.memory.extractor import (
    AssertionMemoryFact,
    InvalidationMemoryFact,
    LLMMemoryFactExtractor,
    MemoryFactExtractor,
    MemoryFactSet,
    StructuredMemoryFactSet,
)
from memory_engine.memory.reconciler import (
    CreateDecision,
    InvalidateDecision,
    MemoryReconciler,
    NoopDecision,
    UpdateDecision,
)
from memory_engine.memory.retriever import RetrievedMemory
from memory_engine.memory.semantic_manager import SemanticMemoryManager
from memory_engine.memory.service import (
    InMemoryMemoryService,
    Memory,
    MemoryEventKind,
    MemoryStatus,
)
from memory_engine.memory.validator import (
    MemoryMutationValidation,
    MemoryMutationValidator,
)
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository


class StaticExtractor(MemoryFactExtractor):
    def __init__(self, facts: MemoryFactSet) -> None:
        self.facts = facts

    async def extract(self, user_message: str) -> MemoryFactSet:
        return self.facts


class ScriptedReconciler(MemoryReconciler):
    def __init__(self, decisions) -> None:
        self.decisions = list(decisions)
        self.calls = []

    async def reconcile(self, fact, candidate_memories):
        self.calls.append((fact, candidate_memories))
        return self.decisions[len(self.calls) - 1]


class BatchOnlyReconciler(MemoryReconciler):
    def __init__(self, decisions=None, error=None) -> None:
        self.decisions = tuple(decisions or ())
        self.error = error
        self.batch_calls = []

    async def reconcile(self, fact, candidate_memories):
        raise AssertionError("Manager must use the batch boundary")

    async def reconcile_many(self, requests):
        self.batch_calls.append(requests)
        if self.error is not None:
            raise self.error
        return self.decisions


class ApprovingValidator(MemoryMutationValidator):
    async def validate(
        self,
        *,
        user_message,
        operation,
        target_memory,
    ) -> MemoryMutationValidation:
        return MemoryMutationValidation(
            approved=True,
            reason="Approved by deterministic test validator.",
        )


class AllActiveRetriever:
    async def retrieve(
        self,
        query,
        candidate_memories,
        limit=5,
    ):
        return tuple(
            RetrievedMemory(memory=memory, score=1.0)
            for memory in candidate_memories[:limit]
        )


class FailingRetriever:
    async def retrieve(self, query, candidate_memories, limit=5):
        raise RuntimeError("embedding service unavailable")


def test_compound_message_processes_each_semantic_fact_independently() -> None:
    user_id = "user-1"
    bakery_goal = Memory(
        id="goal-1",
        content="The user wants to open a bakery.",
        memory_type=MemoryType.GOAL,
        confidence=0.95,
    )
    service = InMemoryMemoryService(
        {user_id: (bakery_goal,)}
    )
    extractor = StaticExtractor(
        MemoryFactSet(
            facts=(
                InvalidationMemoryFact(
                    kind="invalidation",
                    content="The user no longer wants to open a bakery.",
                    memory_type=MemoryType.GOAL,
                    evidence="I gave up opening a bakery",
                    confidence=0.99,
                ),
                AssertionMemoryFact(
                    kind="assertion",
                    content=(
                        "The user prefers receiving feedback face to face."
                    ),
                    memory_type=MemoryType.PREFERENCE,
                    evidence=(
                        "I prefer receiving feedback face to face"
                    ),
                    confidence=0.98,
                ),
            )
        )
    )
    reconciler = ScriptedReconciler(
        [
            InvalidateDecision(
                action="invalidate",
                memory_index=0,
                confidence=0.99,
                explanation="The matching goal was explicitly ended.",
            ),
            CreateDecision(
                action="create",
                confidence=0.98,
                explanation="This is a new durable preference.",
            ),
        ]
    )
    manager = SemanticMemoryManager(
        extractor=extractor,
        reconciler=reconciler,
        memory_service=service,
        memory_retriever=AllActiveRetriever(),
        mutation_validator=ApprovingValidator(),
    )

    changed = asyncio.run(
        manager.capture(
            user_id=user_id,
            user_message=(
                "I gave up opening a bakery and I prefer receiving "
                "feedback face to face"
            ),
        )
    )
    active = asyncio.run(service.list_active(user_id))
    events = asyncio.run(service.list_events(user_id))

    assert len(reconciler.calls) == 2
    assert len(changed) == 2
    assert changed[0].status is MemoryStatus.INVALIDATED
    assert changed[1].memory_type is MemoryType.PREFERENCE
    assert active == (changed[1],)
    assert [event.kind for event in events] == [
        MemoryEventKind.CREATED,
        MemoryEventKind.INVALIDATED,
        MemoryEventKind.CREATED,
    ]
    assert all(
        memory.source_episode_id is not None for memory in changed
    )


def test_unmatched_invalidation_is_noop_but_other_fact_still_creates() -> None:
    service = InMemoryMemoryService()
    reconciler = ScriptedReconciler(
        [
            NoopDecision(
                action="noop",
                confidence=0.99,
                explanation="No matching active goal exists.",
            ),
            CreateDecision(
                action="create",
                confidence=0.98,
                explanation="New preference.",
            ),
        ]
    )
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(
            MemoryFactSet(
                facts=(
                    InvalidationMemoryFact(
                        kind="invalidation",
                        content=(
                            "The user no longer wants to open a bakery."
                        ),
                        memory_type=MemoryType.GOAL,
                        evidence="I gave up opening a bakery",
                        confidence=0.99,
                    ),
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user prefers direct feedback.",
                        memory_type=MemoryType.PREFERENCE,
                        evidence="I prefer direct feedback",
                        confidence=0.98,
                    ),
                )
            )
        ),
        reconciler=reconciler,
        memory_service=service,
        memory_retriever=AllActiveRetriever(),
        mutation_validator=ApprovingValidator(),
    )

    changed = asyncio.run(
        manager.capture(
            "user-1",
            "I gave up opening a bakery and I prefer direct feedback",
        )
    )

    assert len(changed) == 1
    assert changed[0].content == "The user prefers direct feedback."


def test_manager_reconciles_all_prepared_facts_through_one_batch() -> None:
    reconciler = BatchOnlyReconciler(
        decisions=(
            CreateDecision(
                action="create",
                confidence=0.98,
                explanation="New preference.",
            ),
            CreateDecision(
                action="create",
                confidence=0.97,
                explanation="New goal.",
            ),
        )
    )
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(
            MemoryFactSet(
                facts=(
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user prefers direct feedback.",
                        memory_type=MemoryType.PREFERENCE,
                        evidence="I prefer direct feedback",
                        confidence=0.98,
                    ),
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user wants to learn Japanese.",
                        memory_type=MemoryType.GOAL,
                        evidence="I want to learn Japanese",
                        confidence=0.97,
                    ),
                )
            )
        ),
        reconciler=reconciler,
        memory_service=InMemoryMemoryService(),
        memory_retriever=AllActiveRetriever(),
        mutation_validator=ApprovingValidator(),
    )

    changed = asyncio.run(
        manager.capture(
            "user-1",
            "I prefer direct feedback and I want to learn Japanese",
        )
    )

    assert len(reconciler.batch_calls) == 1
    assert len(reconciler.batch_calls[0]) == 2
    assert len(changed) == 2


def test_second_mutation_of_target_changed_earlier_in_batch_is_skipped(
    caplog,
) -> None:
    service = InMemoryMemoryService(
        {
            "user-1": (
                Memory(
                    id="work-preference",
                    content="The user prefers office work.",
                    memory_type=MemoryType.PREFERENCE,
                    confidence=0.95,
                ),
            )
        }
    )
    reconciler = BatchOnlyReconciler(
        decisions=(
            UpdateDecision(
                action="update",
                memory_index=0,
                content="The user prefers remote work.",
                confidence=0.98,
                explanation="Replacement preference.",
            ),
            UpdateDecision(
                action="update",
                memory_index=0,
                content="The user requires fully remote work.",
                confidence=0.97,
                explanation="More specific replacement.",
            ),
        )
    )
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(
            MemoryFactSet(
                facts=(
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user prefers remote work.",
                        memory_type=MemoryType.PREFERENCE,
                        evidence="I prefer remote work",
                        confidence=0.98,
                    ),
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user requires fully remote work.",
                        memory_type=MemoryType.PREFERENCE,
                        evidence="I require fully remote work",
                        confidence=0.97,
                    ),
                )
            )
        ),
        reconciler=reconciler,
        memory_service=service,
        memory_retriever=AllActiveRetriever(),
        mutation_validator=ApprovingValidator(),
    )

    changed = asyncio.run(
        manager.capture(
            "user-1",
            "I prefer remote work and I require fully remote work",
        )
    )

    assert len(changed) == 1
    assert changed[0].content == "The user prefers remote work."
    assert "stale memory target" in caplog.text


def test_batch_reconciliation_failure_applies_no_mutations(caplog) -> None:
    service = InMemoryMemoryService()
    reconciler = BatchOnlyReconciler(
        error=RuntimeError("Reconciliation unavailable.")
    )
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(
            MemoryFactSet(
                facts=(
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user prefers direct feedback.",
                        memory_type=MemoryType.PREFERENCE,
                        evidence="I prefer direct feedback",
                        confidence=0.98,
                    ),
                )
            )
        ),
        reconciler=reconciler,
        memory_service=service,
        memory_retriever=AllActiveRetriever(),
        mutation_validator=ApprovingValidator(),
    )

    changed = asyncio.run(
        manager.capture("user-1", "I prefer direct feedback")
    )

    assert changed == ()
    assert asyncio.run(service.list_active("user-1")) == ()
    assert "failed closed" in caplog.text


def test_retrieval_failure_skips_fact_without_reconciliation(
    caplog,
) -> None:
    service = InMemoryMemoryService()
    reconciler = ScriptedReconciler(
        [
            CreateDecision(
                action="create",
                confidence=0.99,
                explanation="Would create if retrieval succeeded.",
            )
        ]
    )
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(
            MemoryFactSet(
                facts=(
                    AssertionMemoryFact(
                        kind="assertion",
                        content="The user prefers direct feedback.",
                        memory_type=MemoryType.PREFERENCE,
                        evidence="I prefer direct feedback",
                        confidence=0.99,
                    ),
                )
            )
        ),
        reconciler=reconciler,
        memory_service=service,
        memory_retriever=FailingRetriever(),
        mutation_validator=ApprovingValidator(),
    )

    changed = asyncio.run(
        manager.capture("user-1", "I prefer direct feedback")
    )

    assert changed == ()
    assert reconciler.calls == []
    assert asyncio.run(service.list_active("user-1")) == ()
    assert "skipping reconciliation" in caplog.text


def test_extractor_rejects_evidence_not_in_original_message(
    tmp_path,
) -> None:
    prompt_directory = tmp_path / "memory"
    prompt_directory.mkdir()
    (prompt_directory / "extract_facts.txt").write_text(
        "Extract facts.",
        encoding="utf-8",
    )
    response = json.dumps(
        {
            "facts": [
                {
                    "kind": "assertion",
                    "content": "The user prefers direct feedback.",
                    "memory_type": "preference",
                    "evidence": "invented evidence",
                    "confidence": 0.99,
                }
            ]
        }
    )
    extractor = LLMMemoryFactExtractor(
        llm_client=FakeLLMClient(response),
        prompt_repository=PromptRepository(tmp_path),
    )

    with pytest.raises(RuntimeError, match="evidence"):
        asyncio.run(extractor.extract("I prefer direct feedback."))


def test_extraction_provider_schema_avoids_discriminated_one_of() -> None:
    schema = json.dumps(StructuredMemoryFactSet.model_json_schema())

    assert "oneOf" not in schema


def test_fact_set_collapses_only_exact_duplicate_facts() -> None:
    facts = MemoryFactSet(
        facts=(
            AssertionMemoryFact(
                kind="assertion",
                content="The user prefers concise answers.",
                memory_type=MemoryType.PREFERENCE,
                evidence="I prefer concise answers",
                confidence=0.99,
            ),
            AssertionMemoryFact(
                kind="assertion",
                content="  The user prefers concise answers.  ",
                memory_type=MemoryType.PREFERENCE,
                evidence="concise answers are what I prefer",
                confidence=0.97,
            ),
            AssertionMemoryFact(
                kind="assertion",
                content="The user prefers practical examples.",
                memory_type=MemoryType.PREFERENCE,
                evidence="I prefer practical examples",
                confidence=0.98,
            ),
        )
    )

    assert len(facts.facts) == 2
    assert facts.facts[0].evidence == "I prefer concise answers"
    assert facts.facts[1].content == (
        "The user prefers practical examples."
    )
