import asyncio
import json

import pytest
from pydantic import ValidationError

from memory_engine.llm import FakeLLMClient
from memory_engine.memory.extraction_auditor import (
    ExtractionAuditIssue,
    ExtractionAuditIssueType,
    FakeMemoryExtractionAuditor,
    LLMMemoryExtractionAuditor,
    MemoryExtractionAudit,
)
from memory_engine.memory.extractor import AssertionMemoryFact, MemoryFactSet
from memory_engine.memory.reconciler import CreateDecision, MemoryReconciler
from memory_engine.memory.retriever import RetrievedMemory
from memory_engine.memory.semantic_manager import SemanticMemoryManager
from memory_engine.memory.service import InMemoryMemoryService
from memory_engine.memory.validator import MemoryMutationValidator
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository


def _fact(content: str, evidence: str, memory_type=MemoryType.PREFERENCE):
    return AssertionMemoryFact(
        kind="assertion",
        content=content,
        memory_type=memory_type,
        evidence=evidence,
        confidence=0.98,
    )


class StaticExtractor:
    def __init__(self, facts: MemoryFactSet) -> None:
        self.facts = facts

    async def extract(self, user_message: str) -> MemoryFactSet:
        return self.facts


class RecordingReconciler(MemoryReconciler):
    def __init__(self) -> None:
        self.facts = []

    async def reconcile(self, fact, candidate_memories):
        self.facts.append(fact)
        return CreateDecision(
            action="create",
            confidence=0.98,
            explanation="A new durable fact.",
        )


class EmptyRetriever:
    async def retrieve(self, query, candidate_memories, limit=5):
        return tuple(
            RetrievedMemory(memory=item, score=1.0)
            for item in candidate_memories[:limit]
        )


class UnusedValidator(MemoryMutationValidator):
    async def validate(self, *, user_message, operation, target_memory):
        raise AssertionError("CREATE must not call the mutation validator")


def test_audit_verdict_contract_is_strict() -> None:
    with pytest.raises(ValidationError, match="cannot contain issues"):
        MemoryExtractionAudit(
            approved=True,
            issues=(
                ExtractionAuditIssue(
                    issue_type=ExtractionAuditIssueType.MISSING_FACT,
                    description="A fact was missing.",
                ),
            ),
            final_facts=MemoryFactSet(),
        )

    with pytest.raises(ValidationError, match="must explain"):
        MemoryExtractionAudit(
            approved=False,
            final_facts=MemoryFactSet(),
        )


def test_llm_auditor_returns_complete_corrected_fact_set(tmp_path) -> None:
    prompt_directory = tmp_path / "memory"
    prompt_directory.mkdir()
    (prompt_directory / "audit_extraction.txt").write_text(
        "Audit extraction.", encoding="utf-8"
    )
    response = json.dumps(
        {
            "approved": False,
            "issues": [
                {
                    "issue_type": "missing_fact",
                    "description": "The house-saving decision was missing.",
                    "candidate_fact_index": None,
                }
            ],
            "final_facts": {
                "facts": [
                    {
                        "kind": "assertion",
                        "content": "The user prefers remote work.",
                        "memory_type": "preference",
                        "evidence": "I prefer remote work",
                        "confidence": 0.98,
                    },
                    {
                        "kind": "assertion",
                        "content": "The user decided to save for a house.",
                        "memory_type": "decision",
                        "evidence": "I decided to save for a house",
                        "confidence": 0.97,
                    },
                ]
            },
        }
    )
    auditor = LLMMemoryExtractionAuditor(
        FakeLLMClient(response), PromptRepository(tmp_path)
    )

    audit = asyncio.run(
        auditor.audit(
            user_message=(
                "I prefer remote work and I decided to save for a house"
            ),
            candidate_facts=MemoryFactSet(
                facts=(
                    _fact(
                        "The user prefers remote work.",
                        "I prefer remote work",
                    ),
                )
            ),
        )
    )

    assert not audit.approved
    assert len(audit.final_facts.facts) == 2
    assert audit.issues[0].issue_type is ExtractionAuditIssueType.MISSING_FACT


def test_auditor_rejects_corrected_facts_with_invented_evidence(tmp_path) -> None:
    prompt_directory = tmp_path / "memory"
    prompt_directory.mkdir()
    (prompt_directory / "audit_extraction.txt").write_text(
        "Audit extraction.", encoding="utf-8"
    )
    response = json.dumps(
        {
            "approved": False,
            "issues": [
                {
                    "issue_type": "invalid_evidence",
                    "description": "Evidence was corrected."
                }
            ],
            "final_facts": {
                "facts": [
                    {
                        "kind": "assertion",
                        "content": "The user prefers remote work.",
                        "memory_type": "preference",
                        "evidence": "invented evidence",
                        "confidence": 0.98
                    }
                ]
            }
        }
    )
    auditor = LLMMemoryExtractionAuditor(
        FakeLLMClient(response), PromptRepository(tmp_path)
    )

    with pytest.raises(RuntimeError, match="evidence"):
        asyncio.run(
            auditor.audit(
                user_message="I prefer remote work",
                candidate_facts=MemoryFactSet(),
            )
        )


def test_manager_uses_auditors_complete_corrected_set_once() -> None:
    candidate = MemoryFactSet(
        facts=(
            _fact("The user prefers remote work.", "I prefer remote work"),
        )
    )
    corrected = MemoryFactSet(
        facts=(
            candidate.facts[0],
            _fact(
                "The user decided to save for a house.",
                "I decided to save for a house",
                MemoryType.DECISION,
            ),
        )
    )
    auditor = FakeMemoryExtractionAuditor(
        default=MemoryExtractionAudit(
            approved=False,
            issues=(
                ExtractionAuditIssue(
                    issue_type=ExtractionAuditIssueType.MISSING_FACT,
                    description="The decision was missing.",
                ),
            ),
            final_facts=corrected,
        )
    )
    reconciler = RecordingReconciler()
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(candidate),
        extraction_auditor=auditor,
        reconciler=reconciler,
        memory_service=InMemoryMemoryService(),
        memory_retriever=EmptyRetriever(),
        mutation_validator=UnusedValidator(),
    )
    message = "I prefer remote work and I decided to save for a house"

    changed = asyncio.run(manager.capture("user", message))

    assert len(auditor.calls) == 1
    assert reconciler.facts == list(corrected.facts)
    assert len(changed) == 2


def test_manager_fails_closed_when_extraction_audit_fails(caplog) -> None:
    candidate = MemoryFactSet(
        facts=(
            _fact("The user prefers remote work.", "I prefer remote work"),
        )
    )
    reconciler = RecordingReconciler()
    service = InMemoryMemoryService()
    manager = SemanticMemoryManager(
        extractor=StaticExtractor(candidate),
        extraction_auditor=FakeMemoryExtractionAuditor(
            error=RuntimeError("audit unavailable")
        ),
        reconciler=reconciler,
        memory_service=service,
        memory_retriever=EmptyRetriever(),
        mutation_validator=UnusedValidator(),
    )

    changed = asyncio.run(manager.capture("user", "I prefer remote work"))

    assert changed == ()
    assert reconciler.facts == []
    assert asyncio.run(service.list_active("user")) == ()
    assert "audit failed closed" in caplog.text
