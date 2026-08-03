import pytest
from pydantic import ValidationError

from memory_engine.memory.planner import (
    CreateMemoryOperation,
    DeleteMemoryOperation,
    MemoryPlan,
    NoopMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.models import MemoryType


def test_accepts_valid_create_operation() -> None:
    plan = MemoryPlan.model_validate(
        {
            "operations": [
                {
                    "operation": "create",
                    "content": "The user wants to learn Japanese.",
                    "memory_type": "goal",
                    "confidence": 0.95,
                    "explanation": (
                        "The user explicitly stated a durable goal."
                    ),
                }
            ]
        }
    )

    operation = plan.operations[0]

    assert isinstance(operation, CreateMemoryOperation)
    assert operation.content == "The user wants to learn Japanese."
    assert operation.memory_type is MemoryType.GOAL
    assert operation.confidence == 0.95


def test_accepts_valid_update_operation() -> None:
    plan = MemoryPlan.model_validate(
        {
            "operations": [
                {
                    "operation": "update",
                    "memory_index": 0,
                    "content": "The user now lives in Berlin.",
                    "confidence": 0.98,
                    "explanation": (
                        "The user explicitly replaced their previous location."
                    ),
                }
            ]
        }
    )

    operation = plan.operations[0]

    assert isinstance(operation, UpdateMemoryOperation)
    assert operation.memory_index == 0
    assert operation.content == "The user now lives in Berlin."


def test_accepts_valid_delete_operation() -> None:
    plan = MemoryPlan.model_validate(
        {
            "operations": [
                {
                    "operation": "delete",
                    "memory_index": 1,
                    "confidence": 0.99,
                    "explanation": (
                        "The user explicitly said the stored fact is no "
                        "longer true."
                    ),
                }
            ]
        }
    )

    operation = plan.operations[0]

    assert isinstance(operation, DeleteMemoryOperation)
    assert operation.memory_index == 1


def test_accepts_valid_noop_operation() -> None:
    plan = MemoryPlan.model_validate(
        {
            "operations": [
                {
                    "operation": "noop",
                    "confidence": 0.96,
                    "explanation": (
                        "The message contains no durable personal information."
                    ),
                }
            ]
        }
    )

    assert isinstance(plan.operations[0], NoopMemoryOperation)


def test_accepts_delete_and_create_in_same_plan() -> None:
    plan = MemoryPlan.model_validate(
        {
            "operations": [
                {
                    "operation": "delete",
                    "memory_index": 0,
                    "confidence": 0.99,
                    "explanation": (
                        "The user explicitly abandoned the previous goal."
                    ),
                },
                {
                    "operation": "create",
                    "content": (
                        "The user wants to become a better leader."
                    ),
                    "memory_type": "goal",
                    "confidence": 0.97,
                    "explanation": (
                        "The user explicitly stated a new durable goal."
                    ),
                },
            ]
        }
    )

    assert len(plan.operations) == 2
    assert isinstance(plan.operations[0], DeleteMemoryOperation)
    assert isinstance(plan.operations[1], CreateMemoryOperation)


def test_rejects_empty_operation_list() -> None:
    with pytest.raises(
        ValidationError,
        match="must contain at least one operation",
    ):
        MemoryPlan.model_validate({"operations": []})


def test_rejects_unknown_operation() -> None:
    with pytest.raises(ValidationError):
        MemoryPlan.model_validate(
            {
                "operations": [
                    {
                        "operation": "replace",
                        "confidence": 0.90,
                        "explanation": "Unsupported operation.",
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "operation",
    [
        {
            "operation": "create",
            "memory_type": "goal",
            "confidence": 0.95,
            "explanation": "Missing content.",
        },
        {
            "operation": "create",
            "content": "The user wants to learn Japanese.",
            "confidence": 0.95,
            "explanation": "Missing memory type.",
        },
        {
            "operation": "update",
            "content": "The user now lives in Berlin.",
            "confidence": 0.95,
            "explanation": "Missing memory index.",
        },
        {
            "operation": "update",
            "memory_index": 0,
            "confidence": 0.95,
            "explanation": "Missing content.",
        },
        {
            "operation": "delete",
            "confidence": 0.95,
            "explanation": "Missing memory index.",
        },
    ],
    ids=[
        "create-without-content",
        "create-without-memory-type",
        "update-without-memory-index",
        "update-without-content",
        "delete-without-memory-index",
    ],
)
def test_rejects_missing_required_fields(
    operation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MemoryPlan.model_validate({"operations": [operation]})


@pytest.mark.parametrize(
    "operation",
    [
        {
            "operation": "create",
            "memory_index": None,
            "content": "The user wants to learn Japanese.",
            "memory_type": "goal",
            "confidence": 0.95,
            "explanation": "CREATE must omit memory_index.",
        },
        {
            "operation": "delete",
            "memory_index": 0,
            "content": None,
            "confidence": 0.95,
            "explanation": "DELETE must omit content.",
        },
        {
            "operation": "noop",
            "memory_type": None,
            "confidence": 0.95,
            "explanation": "NOOP must omit memory_type.",
        },
    ],
    ids=[
        "create-with-null-memory-index",
        "delete-with-null-content",
        "noop-with-null-memory-type",
    ],
)
def test_rejects_irrelevant_fields(
    operation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryPlan.model_validate({"operations": [operation]})


@pytest.mark.parametrize(
    "operation",
    [
        {
            "operation": "create",
            "content": "",
            "memory_type": "goal",
            "confidence": 0.95,
            "explanation": "Content is blank.",
        },
        {
            "operation": "update",
            "memory_index": 0,
            "content": "",
            "confidence": 0.95,
            "explanation": "Content is blank.",
        },
    ],
    ids=["blank-create-content", "blank-update-content"],
)
def test_rejects_blank_content(
    operation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MemoryPlan.model_validate({"operations": [operation]})


def test_rejects_blank_explanation() -> None:
    with pytest.raises(ValidationError):
        MemoryPlan.model_validate(
            {
                "operations": [
                    {
                        "operation": "noop",
                        "confidence": 0.95,
                        "explanation": "",
                    }
                ]
            }
        )


def test_rejects_negative_memory_index() -> None:
    with pytest.raises(ValidationError):
        MemoryPlan.model_validate(
            {
                "operations": [
                    {
                        "operation": "delete",
                        "memory_index": -1,
                        "confidence": 0.95,
                        "explanation": "An index cannot be negative.",
                    }
                ]
            }
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_rejects_confidence_outside_allowed_range(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        MemoryPlan.model_validate(
            {
                "operations": [
                    {
                        "operation": "noop",
                        "confidence": confidence,
                        "explanation": "Confidence is outside the range.",
                    }
                ]
            }
        )


def test_rejects_noop_combined_with_mutation() -> None:
    with pytest.raises(
        ValidationError,
        match="NOOP cannot be combined",
    ):
        MemoryPlan.model_validate(
            {
                "operations": [
                    {
                        "operation": "noop",
                        "confidence": 0.90,
                        "explanation": "No change is needed.",
                    },
                    {
                        "operation": "create",
                        "content": "The user wants to learn Japanese.",
                        "memory_type": "goal",
                        "confidence": 0.95,
                        "explanation": "A durable goal was stated.",
                    },
                ]
            }
        )


def test_rejects_multiple_mutations_of_same_memory() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot modify the same existing memory more than once",
    ):
        MemoryPlan.model_validate(
            {
                "operations": [
                    {
                        "operation": "update",
                        "memory_index": 0,
                        "content": "The user now lives in Berlin.",
                        "confidence": 0.95,
                        "explanation": "The location changed.",
                    },
                    {
                        "operation": "delete",
                        "memory_index": 0,
                        "confidence": 0.95,
                        "explanation": "The same memory is also deleted.",
                    },
                ]
            }
        )
