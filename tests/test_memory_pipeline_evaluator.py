import pytest

from memory_engine.memory.planner import (
    MemoryPlan,
    MemoryPlanner,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import Memory
from memory_engine.memory.validator import (
    MemoryMutationValidation,
    MemoryMutationValidator,
)
from evaluation.evaluators.memory_pipeline_evaluator import (
    _build_report,
    _content_passes,
    _evaluate_case,
    _score_case,
)
from memory_engine.models import MemoryType


class StaticPlanner(MemoryPlanner):
    async def plan(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> MemoryPlan:
        return MemoryPlan(
            operations=(
                UpdateMemoryOperation(
                    operation="update",
                    memory_index=0,
                    content="The user prefers remote work.",
                    confidence=0.99,
                    explanation="The preference was replaced.",
                ),
            )
        )


class StaticValidator(MemoryMutationValidator):
    def __init__(self, approved: bool) -> None:
        self.approved = approved

    async def validate(
        self,
        *,
        user_message,
        operation,
        target_memory,
    ) -> MemoryMutationValidation:
        return MemoryMutationValidation(
            approved=self.approved,
            reason=(
                "The same work preference is explicitly replaced."
                if self.approved
                else "The proposed update is not safely supported."
            ),
        )


def pipeline_case(*, approved: bool) -> dict:
    expected_content = (
        "remote work"
        if approved
        else "working from an office"
    )
    forbidden_content = (
        "working from an office"
        if approved
        else "remote work"
    )
    return {
        "id": f"pipeline-observability-{approved}",
        "tags": ["update", "observability"],
        "user_message": (
            "I now prefer remote work rather than working "
            "from an office."
        ),
        "initial_memories": [
            {
                "id": "preference-1",
                "content": "The user prefers working from an office.",
                "memory_type": "preference",
                "confidence": 0.95,
            }
        ],
        "expected": {
            "active_memory_count": 1,
            "active_memories": [
                {
                    "id": "preference-1",
                    "memory_type": "preference",
                    "content_assertions": {
                        "must_include": [expected_content],
                        "must_not_include": [forbidden_content],
                    },
                }
            ],
            "deleted_memory_ids": [],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_pipeline_case_records_plan_validation_and_changes(
    approved: bool,
) -> None:
    result = await _evaluate_case(
        planner=StaticPlanner(),
        validator=StaticValidator(approved),
        case=pipeline_case(approved=approved),
        confidence_threshold=0.85,
        retrieval_limit=10,
    )

    assert result.planned_operations[0]["operation"] == "update"
    assert result.validation_decisions == [
        {
            "operation": "update",
            "target_memory_id": "preference-1",
            "approved": approved,
            "reason": (
                "The same work preference is explicitly replaced."
                if approved
                else "The proposed update is not safely supported."
            ),
        }
    ]

    if approved:
        assert result.changed_memories[0]["id"] == "preference-1"
        assert result.changed_memories[0]["content"] == (
            "The user prefers remote work."
        )
    else:
        assert result.changed_memories == []


def creation_case(
    *,
    accepted_memory_types: list[str] | None = None,
) -> dict:
    expected_memory = {
        "memory_type": "goal",
        "content_assertions": {
            "must_include": ["saving", "house"],
            "must_not_include": [],
        },
    }
    if accepted_memory_types is not None:
        expected_memory["accepted_memory_types"] = (
            accepted_memory_types
        )
    return {
        "id": "type-scoring-case",
        "tags": ["create", "taxonomy"],
        "expected": {
            "active_memory_count": 1,
            "active_memories": [expected_memory],
            "deleted_memory_ids": [],
        },
    }


def decision_memory() -> Memory:
    return Memory(
        id="created-1",
        content="The user decided to start saving for a house.",
        memory_type=MemoryType.DECISION,
        confidence=0.95,
    )


def test_accepted_memory_type_counts_as_correct() -> None:
    memory = decision_memory()
    result = _score_case(
        case=creation_case(
            accepted_memory_types=["goal", "decision"]
        ),
        initial_memories=(),
        active_memories=(memory,),
        all_memories=(memory,),
        changed_memories=(memory,),
    )

    assert result.passed
    assert not result.unsafe_final_mutation
    assert not result.unexpected_creation
    assert not result.memory_type_mismatch


def test_type_only_mismatch_is_not_an_unsafe_creation() -> None:
    memory = decision_memory()
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "MemoryManagerPipeline",
    }
    result = _score_case(
        case=creation_case(),
        initial_memories=(),
        active_memories=(memory,),
        all_memories=(memory,),
        changed_memories=(memory,),
    )
    report = _build_report(
        dataset,
        [result],
        model_name="test-model",
    )

    assert not result.passed
    assert not result.unsafe_final_mutation
    assert not result.unexpected_creation
    assert result.memory_type_mismatch
    assert report["summary"]["memory_type_accuracy"] == 0.0


def test_content_assertions_accept_semantic_alternatives() -> None:
    assertions = {
        "must_include_any": [
            ["public speaking"],
            ["speaking in public"],
        ],
        "must_not_include": [],
    }

    assert _content_passes(
        "The user wants to become confident speaking in public.",
        assertions,
    )
