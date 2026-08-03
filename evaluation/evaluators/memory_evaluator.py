from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from memory_engine.dependencies import (
    get_memory_llm,
    get_prompt_repository,
)
from memory_engine.memory.planner import (
    CreateMemoryOperation,
    DeleteMemoryOperation,
    LLMMemoryPlanner,
    MemoryPlan,
    NoopMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import Memory
from memory_engine.models import MemoryType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_evaluation.json"
)


def _parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLMMemoryPlanner against a JSON dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    return parser.parse_args(args)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    schema_valid: bool
    operation_set_correct: bool
    expected_operations: list[dict[str, Any]]
    actual_operations: list[dict[str, Any]]
    target_matches: int
    target_total: int
    create_type_matches: int
    create_type_total: int
    content_matches: int
    content_total: int
    false_create: bool
    false_delete: bool
    false_update: bool
    unsafe_mutation: bool
    duplicate_create: bool
    missed_delete: bool
    error: str | None = None


def _load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_memories(
    records: list[dict[str, Any]],
) -> tuple[Memory, ...]:
    return tuple(
        Memory(
            id=f"evaluation-memory-{index}",
            content=record["content"],
            memory_type=MemoryType(record["memory_type"]),
            confidence=float(record["confidence"]),
        )
        for index, record in enumerate(records)
    )


def _serialize_plan(plan: MemoryPlan) -> list[dict[str, Any]]:
    return [
        operation.model_dump(mode="json")
        for operation in plan.operations
    ]


def _operation_names(
    operations: list[dict[str, Any]],
) -> Counter[str]:
    return Counter(
        operation["operation"]
        for operation in operations
    )


def _multiset_match_count(
    expected_values: list[Any],
    actual_values: list[Any],
) -> int:
    expected = Counter(expected_values)
    actual = Counter(actual_values)
    return sum(
        min(count, actual[value])
        for value, count in expected.items()
    )


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        normalized = f"{token[:-3]}y"
    else:
        normalized = token

    for suffix in ("ing", "ed", "es", "s"):
        if (
            normalized.endswith(suffix)
            and len(normalized) > len(suffix) + 3
        ):
            normalized = normalized[: -len(suffix)]
            break

    # Treat a common silent trailing "e" consistently, so "write" and
    # "writing" normalize to the same form.
    if normalized.endswith("e") and len(normalized) > 4:
        normalized = normalized[:-1]

    return normalized


def _normalized_tokens(text: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(
            r"[a-z0-9]+",
            text.casefold(),
        )
    ]


def _contains_concept(
    content: str,
    concept: str,
) -> bool:
    """
    Match a concept case-insensitively after simple word normalization.

    Concept words must appear in order, but do not need to be adjacent. This
    accepts small wording differences without turning the check into a broad
    semantic judge.
    """

    content_tokens = _normalized_tokens(content)
    concept_tokens = _normalized_tokens(concept)

    if not concept_tokens:
        return True

    position = 0

    for token in content_tokens:
        if token == concept_tokens[position]:
            position += 1

            if position == len(concept_tokens):
                return True

    return False


def _content_passes(
    content: str,
    assertions: dict[str, Any],
) -> bool:
    required_match = all(
        _contains_concept(content, concept)
        for concept in assertions.get("must_include", [])
    )
    alternatives = assertions.get("must_include_any", [])
    alternative_match = (
        not alternatives
        or any(
            all(
                _contains_concept(content, concept)
                for concept in alternative
            )
            for alternative in alternatives
        )
    )
    forbidden_match = all(
        not _contains_concept(content, concept)
        for concept in assertions.get("must_not_include", [])
    )
    return required_match and alternative_match and forbidden_match


def _accepted_memory_types(
    expected: dict[str, Any],
) -> set[str]:
    accepted = expected.get("accepted_memory_types")
    if accepted is not None:
        return set(accepted)
    return {expected["memory_type"]}


def _find_content_candidate(
    expected: dict[str, Any],
    actual_operations: list[dict[str, Any]],
    used_indices: set[int],
) -> tuple[int, dict[str, Any]] | None:
    for index, actual in enumerate(actual_operations):
        if index in used_indices:
            continue

        if actual["operation"] != expected["operation"]:
            continue

        if expected["operation"] == "create":
            if (
                actual.get("memory_type")
                not in _accepted_memory_types(expected)
            ):
                continue

        if expected["operation"] == "update":
            if (
                actual.get("memory_index")
                != expected.get("memory_index")
            ):
                continue

        return index, actual

    return None


def _score_content(
    expected_operations: list[dict[str, Any]],
    actual_operations: list[dict[str, Any]],
) -> tuple[int, int]:
    expected_with_content = [
        operation
        for operation in expected_operations
        if operation["operation"] in {"create", "update"}
    ]
    used_actual_indices: set[int] = set()
    matches = 0

    for expected in expected_with_content:
        candidate = _find_content_candidate(
            expected=expected,
            actual_operations=actual_operations,
            used_indices=used_actual_indices,
        )

        if candidate is None:
            continue

        actual_index, actual = candidate
        used_actual_indices.add(actual_index)

        content = actual.get("content")

        if (
            isinstance(content, str)
            and _content_passes(
                content,
                expected["content_assertions"],
            )
        ):
            matches += 1

    return matches, len(expected_with_content)


def _score_create_types(
    expected_operations: list[dict[str, Any]],
    actual_operations: list[dict[str, Any]],
) -> tuple[int, int]:
    expected_creates = [
        operation
        for operation in expected_operations
        if operation["operation"] == "create"
    ]
    actual_creates = [
        operation
        for operation in actual_operations
        if operation["operation"] == "create"
    ]
    used_actual_indices: set[int] = set()
    matches = 0

    for expected in expected_creates:
        accepted_types = _accepted_memory_types(expected)

        for index, actual in enumerate(actual_creates):
            if index in used_actual_indices:
                continue

            if actual.get("memory_type") not in accepted_types:
                continue

            matches += 1
            used_actual_indices.add(index)
            break

    return matches, len(expected_creates)


def _score_case(
    case: dict[str, Any],
    actual_operations: list[dict[str, Any]],
    *,
    schema_valid: bool,
    error: str | None = None,
) -> CaseResult:
    expected_operations = case["expected"]["operations"]

    operation_set_correct = (
        _operation_names(expected_operations)
        == _operation_names(actual_operations)
    )

    expected_targets = [
        (
            operation["operation"],
            operation["memory_index"],
        )
        for operation in expected_operations
        if operation["operation"] in {"update", "delete"}
    ]
    actual_targets = [
        (
            operation["operation"],
            operation["memory_index"],
        )
        for operation in actual_operations
        if operation["operation"] in {"update", "delete"}
    ]
    target_matches = _multiset_match_count(
        expected_targets,
        actual_targets,
    )

    actual_create_types = [
        operation["memory_type"]
        for operation in actual_operations
        if operation["operation"] == "create"
    ]
    create_type_matches, create_type_total = _score_create_types(
        expected_operations=expected_operations,
        actual_operations=actual_operations,
    )

    content_matches, content_total = _score_content(
        expected_operations=expected_operations,
        actual_operations=actual_operations,
    )

    expects_create = create_type_total > 0
    produced_create = bool(actual_create_types)
    false_create = produced_create and not expects_create

    expected_deletes = [
        target
        for target in expected_targets
        if target[0] == "delete"
    ]
    actual_deletes = [
        target
        for target in actual_targets
        if target[0] == "delete"
    ]
    expected_updates = [
        target
        for target in expected_targets
        if target[0] == "update"
    ]
    actual_updates = [
        target
        for target in actual_targets
        if target[0] == "update"
    ]
    matched_deletes = _multiset_match_count(
        expected_deletes,
        actual_deletes,
    )
    matched_mutations = _multiset_match_count(
        expected_targets,
        actual_targets,
    )
    false_delete = bool(actual_deletes) and not expected_deletes
    false_update = bool(actual_updates) and not expected_updates
    unsafe_mutation = matched_mutations < len(actual_targets)
    duplicate_create = (
        "duplicate" in case["tags"]
        and produced_create
    )
    missed_delete = (
        bool(expected_deletes)
        and matched_deletes < len(expected_deletes)
    )

    targets_correct = target_matches == len(expected_targets)
    create_types_correct = (
        create_type_matches == create_type_total
    )
    content_correct = content_matches == content_total

    return CaseResult(
        case_id=case["id"],
        passed=(
            schema_valid
            and operation_set_correct
            and targets_correct
            and create_types_correct
            and content_correct
            and not false_create
            and not false_delete
            and not false_update
            and not unsafe_mutation
            and not duplicate_create
            and not missed_delete
        ),
        schema_valid=schema_valid,
        operation_set_correct=operation_set_correct,
        expected_operations=expected_operations,
        actual_operations=actual_operations,
        target_matches=target_matches,
        target_total=len(expected_targets),
        create_type_matches=create_type_matches,
        create_type_total=create_type_total,
        content_matches=content_matches,
        content_total=content_total,
        false_create=false_create,
        false_delete=false_delete,
        false_update=false_update,
        unsafe_mutation=unsafe_mutation,
        duplicate_create=duplicate_create,
        missed_delete=missed_delete,
        error=error,
    )


async def _evaluate_case(
    planner: LLMMemoryPlanner,
    case: dict[str, Any],
) -> CaseResult:
    memories = _build_memories(case["active_memories"])

    try:
        plan = await planner.plan(
            user_message=case["user_message"],
            active_memories=memories,
        )
    except ValueError as error:
        # The JSON matched MemoryPlan, but a temporary index was outside the
        # supplied memory collection.
        return _score_case(
            case,
            [],
            schema_valid=True,
            error=str(error),
        )
    except RuntimeError as error:
        if "invalid structured response" not in str(error):
            # Connection, timeout, and Ollama HTTP failures are infrastructure
            # failures. They must not be misreported as model-quality results.
            raise

        return _score_case(
            case,
            [],
            schema_valid=False,
            error=str(error),
        )

    return _score_case(
        case,
        _serialize_plan(plan),
        schema_valid=True,
    )


def _rate(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def _build_report(
    dataset: dict[str, Any],
    results: list[CaseResult],
) -> dict[str, Any]:
    case_count = len(results)

    target_matches = sum(
        result.target_matches
        for result in results
    )
    target_total = sum(
        result.target_total
        for result in results
    )
    create_type_matches = sum(
        result.create_type_matches
        for result in results
    )
    create_type_total = sum(
        result.create_type_total
        for result in results
    )
    content_matches = sum(
        result.content_matches
        for result in results
    )
    content_total = sum(
        result.content_total
        for result in results
    )

    false_create_eligible = sum(
        not any(
            operation["operation"] == "create"
            for operation in result.expected_operations
        )
        for result in results
    )
    false_delete_eligible = sum(
        not any(
            operation["operation"] == "delete"
            for operation in result.expected_operations
        )
        for result in results
    )
    false_update_eligible = sum(
        not any(
            operation["operation"] == "update"
            for operation in result.expected_operations
        )
        for result in results
    )
    duplicate_create_eligible = sum(
        "duplicate" in case["tags"]
        for case in dataset["cases"]
    )
    delete_eligible = sum(
        any(
            operation["operation"] == "delete"
            for operation in result.expected_operations
        )
        for result in results
    )

    return {
        "schema_version": "1.0",
        "dataset_schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "summary": {
            "case_count": case_count,
            "passed_cases": sum(
                result.passed
                for result in results
            ),
            "schema_valid_response_rate": _rate(
                sum(result.schema_valid for result in results),
                case_count,
            ),
            "correct_operation_set_rate": _rate(
                sum(
                    result.operation_set_correct
                    for result in results
                ),
                case_count,
            ),
            "correct_target_index_rate": _rate(
                target_matches,
                target_total,
            ),
            "correct_create_memory_type_rate": _rate(
                create_type_matches,
                create_type_total,
            ),
            "content_faithfulness_rate": _rate(
                content_matches,
                content_total,
            ),
            "false_create_rate": _rate(
                sum(result.false_create for result in results),
                false_create_eligible,
            ),
            "false_delete_rate": _rate(
                sum(result.false_delete for result in results),
                false_delete_eligible,
            ),
            "false_update_rate": _rate(
                sum(result.false_update for result in results),
                false_update_eligible,
            ),
            "unsafe_mutation_rate": _rate(
                sum(result.unsafe_mutation for result in results),
                case_count,
            ),
            "duplicate_create_rate": _rate(
                sum(result.duplicate_create for result in results),
                duplicate_create_eligible,
            ),
            "missed_delete_rate": _rate(
                sum(result.missed_delete for result in results),
                delete_eligible,
            ),
        },
        "cases": [
            asdict(result)
            for result in results
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]

    print("\nMemory planner evaluation")
    print("=" * 40)
    print(
        f"Cases passed: "
        f"{summary['passed_cases']}/{summary['case_count']}"
    )

    for metric in (
        "schema_valid_response_rate",
        "correct_operation_set_rate",
        "correct_target_index_rate",
        "correct_create_memory_type_rate",
        "content_faithfulness_rate",
        "false_create_rate",
        "false_delete_rate",
        "false_update_rate",
        "unsafe_mutation_rate",
        "duplicate_create_rate",
        "missed_delete_rate",
    ):
        value = summary[metric]
        formatted = "n/a" if value is None else f"{value:.1%}"
        print(f"{metric}: {formatted}")

    failures = [
        case
        for case in report["cases"]
        if not case["passed"]
    ]

    if failures:
        print("\nFailed cases")
        print("-" * 40)

        for case in failures:
            print(f"- {case['case_id']}")

            if case["error"]:
                print(f"  error: {case['error']}")

            print(
                "  expected: "
                f"{case['expected_operations']}"
            )
            print(
                "  actual:   "
                f"{case['actual_operations']}"
            )


async def evaluate(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    planner = LLMMemoryPlanner(
        llm_client=get_memory_llm(),
        prompt_repository=get_prompt_repository(),
    )

    results: list[CaseResult] = []

    for position, case in enumerate(
        dataset["cases"],
        start=1,
    ):
        print(
            f"[{position}/{len(dataset['cases'])}] "
            f"{case['id']}"
        )
        results.append(
            await _evaluate_case(
                planner=planner,
                case=case,
            )
        )

    report = _build_report(
        dataset=dataset,
        results=results,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _print_report(report)
    print(f"\nReport written to: {report_path}")

    return report


def main(args: list[str] | None = None) -> None:
    parsed = _parse_args(args)
    asyncio.run(
        evaluate(
            dataset_path=parsed.dataset,
            report_path=parsed.report,
        )
    )


if __name__ == "__main__":
    main()
