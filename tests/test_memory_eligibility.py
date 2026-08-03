import asyncio
import json

import pytest
from pydantic import ValidationError

from memory_engine.llm import FakeLLMClient
from memory_engine.memory.eligibility import (
    FakeMemoryEligibilityGate,
    LLMMemoryEligibilityGate,
    MemoryEligibilityAssessment,
    MemoryEligibilityReason,
)
from memory_engine.memory.extraction_auditor import FakeMemoryExtractionAuditor
from memory_engine.memory.extractor import MemoryFactSet
from memory_engine.memory.semantic_manager import SemanticMemoryManager
from memory_engine.memory.service import InMemoryMemoryService
from memory_engine.prompting import PromptRepository


class FailingExtractor:
    async def extract(self, user_message):
        raise AssertionError("Ineligible interaction reached extraction")


class FailingCollaborator:
    def __getattr__(self, name):
        raise AssertionError(f"Ineligible interaction reached {name}")


def _assessment(eligible: bool) -> MemoryEligibilityAssessment:
    return MemoryEligibilityAssessment(
        eligible=eligible,
        reason=(
            MemoryEligibilityReason.POSSIBLE_DURABLE_INFORMATION
            if eligible
            else MemoryEligibilityReason.NO_DURABLE_INFORMATION
        ),
        confidence=0.98,
        explanation="Deterministic test assessment.",
    )


def _manager(gate, service):
    return SemanticMemoryManager(
        extractor=FailingExtractor(),
        extraction_auditor=FakeMemoryExtractionAuditor(
            error=AssertionError("Auditor should not be called")
        ),
        eligibility_gate=gate,
        reconciler=FailingCollaborator(),
        memory_service=service,
        memory_retriever=FailingCollaborator(),
        mutation_validator=FailingCollaborator(),
    )


def test_eligibility_contract_rejects_inconsistent_reason() -> None:
    with pytest.raises(ValidationError, match="eligible reason"):
        MemoryEligibilityAssessment(
            eligible=True,
            reason=MemoryEligibilityReason.NO_DURABLE_INFORMATION,
            confidence=0.9,
            explanation="Inconsistent.",
        )


def test_ineligible_interaction_skips_episode_extraction_and_audit() -> None:
    service = InMemoryMemoryService()
    gate = FakeMemoryEligibilityGate(default=_assessment(False))
    manager = _manager(gate, service)

    changed = asyncio.run(manager.capture("user", "Hello, how are you?"))

    assert changed == ()
    assert gate.calls == ["Hello, how are you?"]
    assert service._episodes_by_user == {}


def test_eligibility_failure_fails_closed_before_episode(caplog) -> None:
    service = InMemoryMemoryService()
    manager = _manager(
        FakeMemoryEligibilityGate(error=RuntimeError("model unavailable")),
        service,
    )

    assert asyncio.run(manager.capture("user", "I live in Berlin")) == ()
    assert service._episodes_by_user == {}
    assert "eligibility assessment failed closed" in caplog.text


def test_llm_eligibility_gate_uses_validated_structured_output(tmp_path) -> None:
    prompt_directory = tmp_path / "memory"
    prompt_directory.mkdir()
    (prompt_directory / "assess_eligibility.txt").write_text(
        "Assess eligibility.", encoding="utf-8"
    )
    gate = LLMMemoryEligibilityGate(
        FakeLLMClient(
            json.dumps(
                {
                    "eligible": True,
                    "reason": "possible_durable_information",
                    "confidence": 0.97,
                    "explanation": "A durable location fact is possible.",
                }
            )
        ),
        PromptRepository(tmp_path),
    )

    result = asyncio.run(gate.assess("I live in Berlin"))

    assert result.eligible
    assert result.reason is MemoryEligibilityReason.POSSIBLE_DURABLE_INFORMATION
