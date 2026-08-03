from evaluation.evaluators.memory_evaluator import (
    _build_report,
    _score_case,
)


def _case(
    *,
    tags: list[str] | None = None,
    expected_operations: list[dict] | None = None,
) -> dict:
    return {
        "id": "metric-case",
        "tags": tags or ["safety"],
        "active_memories": [
            {
                "content": "The user prefers working in the morning.",
                "memory_type": "preference",
                "confidence": 0.9,
            }
        ],
        "expected": {
            "operations": expected_operations
            or [{"operation": "noop"}],
        },
    }


def test_false_and_unsafe_mutations_are_scored() -> None:
    case = _case()

    false_delete = _score_case(
        case,
        [{"operation": "delete", "memory_index": 0}],
        schema_valid=True,
    )
    false_update = _score_case(
        case,
        [
            {
                "operation": "update",
                "memory_index": 0,
                "content": "The user prefers working at night.",
            }
        ],
        schema_valid=True,
    )

    assert false_delete.false_delete
    assert false_delete.unsafe_mutation
    assert false_update.false_update
    assert false_update.unsafe_mutation


def test_wrong_mutation_target_is_unsafe() -> None:
    case = _case(
        expected_operations=[
            {
                "operation": "update",
                "memory_index": 0,
                "content_assertions": {
                    "must_include": ["morning"],
                    "must_not_include": [],
                },
            }
        ]
    )

    result = _score_case(
        case,
        [
            {
                "operation": "update",
                "memory_index": 1,
                "content": "The user prefers working in the morning.",
            }
        ],
        schema_valid=True,
    )

    assert not result.false_update
    assert result.unsafe_mutation


def test_duplicate_create_rate_uses_tagged_cases() -> None:
    case = _case(tags=["noop", "duplicate"])
    result = _score_case(
        case,
        [
            {
                "operation": "create",
                "memory_type": "preference",
                "content": "The user prefers working in the morning.",
            }
        ],
        schema_valid=True,
    )
    report = _build_report(
        {
            "schema_version": "1.0",
            "evaluation_target": "LLMMemoryPlanner",
            "cases": [case],
        },
        [result],
    )

    assert result.duplicate_create
    assert report["summary"]["duplicate_create_rate"] == 1.0
