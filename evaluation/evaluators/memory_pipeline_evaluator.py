from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from memory_engine.config import get_settings
from memory_engine.dependencies import get_memory_llm, get_prompt_repository
from memory_engine.memory.manager import MemoryManager
from memory_engine.memory.planner import (
    DeleteMemoryOperation,
    LLMMemoryPlanner,
    MemoryPlan,
    MemoryPlanner,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import (
    Memory,
    MemoryService,
    MemoryStatus,
)
from memory_engine.memory.validator import (
    LLMMemoryMutationValidator,
    MemoryMutationValidation,
    MemoryMutationValidator,
)
from memory_engine.models import MemoryCandidate, MemoryType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_pipeline.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_pipeline_evaluation.json"
)
USER_ID = "pipeline-evaluation-user"


class RecordingMemoryPlanner(MemoryPlanner):
    def __init__(self, delegate: MemoryPlanner) -> None:
        self.delegate = delegate
        self.planned_operations: list[dict[str, Any]] = []

    async def plan(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> MemoryPlan:
        plan = await self.delegate.plan(
            user_message=user_message,
            active_memories=active_memories,
        )
        self.planned_operations = [
            operation.model_dump(mode="json")
            for operation in plan.operations
        ]
        return plan


class RecordingMemoryMutationValidator(MemoryMutationValidator):
    def __init__(
        self,
        delegate: MemoryMutationValidator,
    ) -> None:
        self.delegate = delegate
        self.validation_decisions: list[dict[str, Any]] = []

    async def validate(
        self,
        *,
        user_message: str,
        operation: UpdateMemoryOperation | DeleteMemoryOperation,
        target_memory: Memory,
    ) -> MemoryMutationValidation:
        try:
            result = await self.delegate.validate(
                user_message=user_message,
                operation=operation,
                target_memory=target_memory,
            )
        except Exception as error:
            self.validation_decisions.append(
                {
                    "operation": operation.operation,
                    "target_memory_id": target_memory.id,
                    "approved": False,
                    "reason": (
                        "Validator error: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )
            raise

        self.validation_decisions.append(
            {
                "operation": operation.operation,
                "target_memory_id": target_memory.id,
                "approved": result.approved,
                "reason": result.reason,
            }
        )
        return result


class OracleMemoryService(MemoryService):
    """
    Evaluation-only service whose search returns active memories in seed order.

    Retrieval quality is evaluated separately. This service keeps the pipeline
    evaluation focused on planning, mutation validation, execution, and final
    state.
    """

    def __init__(self, memories: tuple[Memory, ...]) -> None:
        self._memories = list(memories)
        self._created_count = 0

    async def list_active(
        self,
        user_id: str,
    ) -> tuple[Memory, ...]:
        return tuple(
            memory
            for memory in self._memories
            if memory.status == MemoryStatus.ACTIVE
        )

    async def list_all(
        self,
        user_id: str,
    ) -> tuple[Memory, ...]:
        return tuple(self._memories)

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> tuple[Memory, ...]:
        if limit <= 0:
            return ()

        active = await self.list_active(user_id=user_id)
        return active[:limit]

    async def save(
        self,
        user_id: str,
        candidate: MemoryCandidate,
    ) -> Memory:
        self._created_count += 1
        memory = Memory(
            id=f"evaluation-created-{self._created_count}",
            content=candidate.content,
            memory_type=candidate.memory_type,
            confidence=candidate.confidence,
        )
        self._memories.append(memory)
        return memory

    async def update(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        confidence: float,
    ) -> Memory:
        for index, memory in enumerate(self._memories):
            if (
                memory.id == memory_id
                and memory.status == MemoryStatus.ACTIVE
            ):
                updated = Memory(
                    id=memory.id,
                    content=content,
                    memory_type=memory.memory_type,
                    confidence=confidence,
                    status=MemoryStatus.ACTIVE,
                )
                self._memories[index] = updated
                return updated

        raise ValueError(f"Active memory not found: {memory_id}")

    async def delete(
        self,
        user_id: str,
        memory_id: str,
    ) -> Memory:
        for index, memory in enumerate(self._memories):
            if (
                memory.id == memory_id
                and memory.status == MemoryStatus.ACTIVE
            ):
                deleted = Memory(
                    id=memory.id,
                    content=memory.content,
                    memory_type=memory.memory_type,
                    confidence=memory.confidence,
                    status=MemoryStatus.DELETED,
                )
                self._memories[index] = deleted
                return deleted

        raise ValueError(f"Active memory not found: {memory_id}")


@dataclass(frozen=True)
class PipelineCaseResult:
    case_id: str
    passed: bool
    expected_active_memory_count: int
    actual_active_memory_count: int
    expected_active_matches: int
    expected_active_total: int
    expected_deleted_ids: list[str]
    actual_deleted_ids: list[str]
    expected_change_matches: int
    expected_change_total: int
    unsafe_final_mutation: bool
    incorrect_deletion: bool
    incorrect_update: bool
    unexpected_creation: bool
    duplicate_active_memory: bool
    unnecessary_update: bool
    memory_type_matches: int
    memory_type_total: int
    memory_type_mismatch: bool
    planned_operations: list[dict[str, Any]]
    validation_decisions: list[dict[str, Any]]
    final_active_memories: list[dict[str, Any]]
    changed_memories: list[dict[str, Any]]
    error: str | None = None


def _build_initial_memories(
    records: list[dict[str, Any]],
) -> tuple[Memory, ...]:
    return tuple(
        Memory(
            id=record["id"],
            content=record["content"],
            memory_type=MemoryType(record["memory_type"]),
            confidence=float(record["confidence"]),
        )
        for record in records
    )


def _serialize_memory(memory: Memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "content": memory.content,
        "memory_type": memory.memory_type.value,
        "confidence": memory.confidence,
        "status": memory.status.value,
    }


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        normalized = f"{token[:-3]}y"
    else:
        normalized = token

    for suffix in ("ing", "ed", "es", "ly", "s"):
        if (
            normalized.endswith(suffix)
            and len(normalized) > len(suffix) + 3
        ):
            normalized = normalized[: -len(suffix)]
            break

    if normalized.endswith("e") and len(normalized) > 4:
        normalized = normalized[:-1]

    return normalized


def _normalized_tokens(text: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
    ]


def _contains_required_concept(
    content: str,
    concept: str,
) -> bool:
    content_tokens = Counter(_normalized_tokens(content))
    concept_tokens = Counter(_normalized_tokens(concept))

    return all(
        content_tokens[token] >= count
        for token, count in concept_tokens.items()
    )


def _contains_forbidden_claim(
    content: str,
    claim: str,
) -> bool:
    content_tokens = _normalized_tokens(content)
    claim_tokens = _normalized_tokens(claim)

    if not claim_tokens:
        return False

    length = len(claim_tokens)
    return any(
        content_tokens[start : start + length] == claim_tokens
        for start in range(len(content_tokens) - length + 1)
    )


def _content_passes(
    content: str,
    assertions: dict[str, Any],
) -> bool:
    required_match = all(
        _contains_required_concept(content, concept)
        for concept in assertions.get("must_include", [])
    )
    alternatives = assertions.get("must_include_any", [])
    alternative_match = (
        not alternatives
        or any(
            all(
                _contains_required_concept(content, concept)
                for concept in alternative
            )
            for alternative in alternatives
        )
    )
    forbidden_match = all(
        not _contains_forbidden_claim(content, claim)
        for claim in assertions.get("must_not_include", [])
    )
    return required_match and alternative_match and forbidden_match


def _accepted_memory_types(
    expected: dict[str, Any],
) -> set[str]:
    accepted = expected.get("accepted_memory_types")
    if accepted is not None:
        return set(accepted)
    return {expected["memory_type"]}


def _memory_identity_and_content_match(
    memory: Memory,
    expected: dict[str, Any],
) -> bool:
    expected_id = expected.get("id")
    return (
        (expected_id is None or memory.id == expected_id)
        and _content_passes(
            memory.content,
            expected["content_assertions"],
        )
    )


def _memory_matches_expected(
    memory: Memory,
    expected: dict[str, Any],
) -> bool:
    return (
        _memory_identity_and_content_match(memory, expected)
        and memory.memory_type.value
        in _accepted_memory_types(expected)
    )


def _match_expected_active(
    expected_memories: list[dict[str, Any]],
    active_memories: tuple[Memory, ...],
) -> tuple[int, set[int]]:
    matches = 0
    used_actual_indices: set[int] = set()

    for expected in expected_memories:
        for index, actual in enumerate(active_memories):
            if index in used_actual_indices:
                continue

            if _memory_matches_expected(actual, expected):
                matches += 1
                used_actual_indices.add(index)
                break

    return matches, used_actual_indices


def _match_expected_semantically(
    expected_memories: list[dict[str, Any]],
    active_memories: tuple[Memory, ...],
) -> tuple[int, set[int], int]:
    """
    Match identity and content without treating taxonomy as safety.

    Returns semantic matches, used actual indices, and the number whose type
    is one of the expected memory's accepted types.
    """

    matches = 0
    type_matches = 0
    used_actual_indices: set[int] = set()

    for expected in expected_memories:
        for index, actual in enumerate(active_memories):
            if index in used_actual_indices:
                continue
            if not _memory_identity_and_content_match(actual, expected):
                continue

            matches += 1
            used_actual_indices.add(index)
            if (
                actual.memory_type.value
                in _accepted_memory_types(expected)
            ):
                type_matches += 1
            break

    return matches, used_actual_indices, type_matches


def _expected_changes(
    initial_memories: tuple[Memory, ...],
    expected_active: list[dict[str, Any]],
    expected_deleted_ids: set[str],
) -> list[dict[str, Any]]:
    initial_by_id = {
        memory.id: memory
        for memory in initial_memories
    }
    changes: list[dict[str, Any]] = [
        {
            "kind": "delete",
            "memory_id": memory_id,
        }
        for memory_id in expected_deleted_ids
    ]

    for expected in expected_active:
        expected_id = expected.get("id")

        if expected_id is None:
            changes.append(
                {
                    "kind": "create",
                    "expected": expected,
                }
            )
            continue

        initial = initial_by_id.get(expected_id)

        if (
            initial is not None
            and not _memory_matches_expected(initial, expected)
        ):
            changes.append(
                {
                    "kind": "update",
                    "expected": expected,
                }
            )

    return changes


def _score_expected_changes(
    changes: list[dict[str, Any]],
    active_memories: tuple[Memory, ...],
    actual_deleted_ids: set[str],
) -> int:
    matches = 0
    used_active_indices: set[int] = set()

    for change in changes:
        if change["kind"] == "delete":
            if change["memory_id"] in actual_deleted_ids:
                matches += 1
            continue

        expected = change["expected"]

        for index, memory in enumerate(active_memories):
            if index in used_active_indices:
                continue

            if _memory_identity_and_content_match(memory, expected):
                matches += 1
                used_active_indices.add(index)
                break

    return matches


def _has_exact_active_duplicates(
    active_memories: tuple[Memory, ...],
) -> bool:
    normalized = [
        (
            memory.memory_type.value,
            memory.content.strip().casefold(),
        )
        for memory in active_memories
    ]
    return len(normalized) != len(set(normalized))


def _score_case(
    case: dict[str, Any],
    initial_memories: tuple[Memory, ...],
    active_memories: tuple[Memory, ...],
    all_memories: tuple[Memory, ...],
    changed_memories: tuple[Memory, ...],
    planned_operations: list[dict[str, Any]] | None = None,
    validation_decisions: list[dict[str, Any]] | None = None,
    *,
    error: str | None = None,
) -> PipelineCaseResult:
    expected = case["expected"]
    expected_active = expected["active_memories"]
    expected_deleted_ids = set(expected["deleted_memory_ids"])
    actual_deleted_ids = {
        memory.id
        for memory in all_memories
        if memory.status == MemoryStatus.DELETED
    }

    active_matches, _ = _match_expected_active(
        expected_memories=expected_active,
        active_memories=active_memories,
    )
    (
        _,
        semantically_matched_indices,
        _,
    ) = _match_expected_semantically(
        expected_memories=expected_active,
        active_memories=active_memories,
    )

    expected_changes = _expected_changes(
        initial_memories=initial_memories,
        expected_active=expected_active,
        expected_deleted_ids=expected_deleted_ids,
    )
    expected_change_matches = _score_expected_changes(
        changes=expected_changes,
        active_memories=active_memories,
        actual_deleted_ids=actual_deleted_ids,
    )
    expected_typed_changes = [
        change["expected"]
        for change in expected_changes
        if change["kind"] in {"create", "update"}
    ]
    (
        memory_type_total,
        _,
        memory_type_matches,
    ) = _match_expected_semantically(
        expected_memories=expected_typed_changes,
        active_memories=active_memories,
    )

    initial_ids = {
        memory.id
        for memory in initial_memories
    }
    expected_preserved_ids = {
        memory["id"]
        for memory in expected_active
        if memory.get("id") in initial_ids
    }
    actual_active_by_id = {
        memory.id: memory
        for memory in active_memories
    }
    expected_active_by_id = {
        memory["id"]: memory
        for memory in expected_active
        if "id" in memory
    }
    initial_by_id = {
        memory.id: memory
        for memory in initial_memories
    }

    incorrect_deletion = bool(
        (actual_deleted_ids - expected_deleted_ids)
        | (expected_preserved_ids - actual_active_by_id.keys())
    )

    incorrect_update = any(
        (
            actual_memory.content
            != initial_by_id[memory_id].content
            or actual_memory.memory_type
            != initial_by_id[memory_id].memory_type
        )
        and (
            memory_id not in expected_active_by_id
            or not _memory_identity_and_content_match(
                actual_memory,
                expected_active_by_id[memory_id],
            )
        )
        for memory_id, actual_memory in actual_active_by_id.items()
        if memory_id in initial_by_id
    )

    unexpected_creation = any(
        memory.id not in initial_ids
        and index not in semantically_matched_indices
        for index, memory in enumerate(active_memories)
    )
    memory_type_mismatch = (
        memory_type_matches < memory_type_total
    )

    duplicate_active_memory = (
        _has_exact_active_duplicates(active_memories)
        or (
            "duplicate" in case["tags"]
            and len(active_memories)
            > expected["active_memory_count"]
        )
    )

    unnecessary_update = any(
        memory_id in expected_active_by_id
        and memory_id in actual_active_by_id
        and _memory_matches_expected(
            initial_memory,
            expected_active_by_id[memory_id],
        )
        and _memory_identity_and_content_match(
            actual_active_by_id[memory_id],
            expected_active_by_id[memory_id],
        )
        and (
            actual_active_by_id[memory_id].content
            != initial_memory.content
            or actual_active_by_id[memory_id].memory_type
            != initial_memory.memory_type
        )
        for memory_id, initial_memory in initial_by_id.items()
    )

    unsafe_final_mutation = (
        incorrect_deletion
        or incorrect_update
        or unexpected_creation
        or duplicate_active_memory
    )

    exact_final_state = (
        error is None
        and len(active_memories)
        == expected["active_memory_count"]
        and active_matches == len(expected_active)
        and actual_deleted_ids == expected_deleted_ids
        and not unsafe_final_mutation
    )

    return PipelineCaseResult(
        case_id=case["id"],
        passed=exact_final_state,
        expected_active_memory_count=expected["active_memory_count"],
        actual_active_memory_count=len(active_memories),
        expected_active_matches=active_matches,
        expected_active_total=len(expected_active),
        expected_deleted_ids=sorted(expected_deleted_ids),
        actual_deleted_ids=sorted(actual_deleted_ids),
        expected_change_matches=expected_change_matches,
        expected_change_total=len(expected_changes),
        unsafe_final_mutation=unsafe_final_mutation,
        incorrect_deletion=incorrect_deletion,
        incorrect_update=incorrect_update,
        unexpected_creation=unexpected_creation,
        duplicate_active_memory=duplicate_active_memory,
        unnecessary_update=unnecessary_update,
        memory_type_matches=memory_type_matches,
        memory_type_total=memory_type_total,
        memory_type_mismatch=memory_type_mismatch,
        planned_operations=planned_operations or [],
        validation_decisions=validation_decisions or [],
        final_active_memories=[
            _serialize_memory(memory)
            for memory in active_memories
        ],
        changed_memories=[
            _serialize_memory(memory)
            for memory in changed_memories
        ],
        error=error,
    )


async def _evaluate_case(
    planner: MemoryPlanner,
    validator: MemoryMutationValidator,
    case: dict[str, Any],
    *,
    confidence_threshold: float,
    retrieval_limit: int,
) -> PipelineCaseResult:
    initial_memories = _build_initial_memories(
        case["initial_memories"]
    )
    service = OracleMemoryService(initial_memories)
    recording_planner = RecordingMemoryPlanner(planner)
    recording_validator = RecordingMemoryMutationValidator(
        validator
    )
    manager = MemoryManager(
        planner=recording_planner,
        memory_service=service,
        mutation_validator=recording_validator,
        confidence_threshold=confidence_threshold,
        retrieval_limit=retrieval_limit,
    )

    try:
        changed_memories = await manager.capture(
            user_id=USER_ID,
            user_message=case["user_message"],
        )
        error = None
    except ValueError as exception:
        changed_memories = ()
        error = str(exception)
    except RuntimeError as exception:
        if "invalid structured response" not in str(exception):
            raise

        changed_memories = ()
        error = str(exception)

    active_memories = await service.list_active(user_id=USER_ID)
    all_memories = await service.list_all(user_id=USER_ID)

    return _score_case(
        case=case,
        initial_memories=initial_memories,
        active_memories=active_memories,
        all_memories=all_memories,
        changed_memories=changed_memories,
        planned_operations=recording_planner.planned_operations,
        validation_decisions=(
            recording_validator.validation_decisions
        ),
        error=error,
    )


def _rate(
    numerator: int,
    denominator: int,
) -> float | None:
    return numerator / denominator if denominator else None


def _build_report(
    dataset: dict[str, Any],
    results: list[PipelineCaseResult],
    *,
    model_name: str,
) -> dict[str, Any]:
    case_count = len(results)
    expected_change_matches = sum(
        result.expected_change_matches
        for result in results
    )
    expected_change_total = sum(
        result.expected_change_total
        for result in results
    )
    memory_type_matches = sum(
        result.memory_type_matches
        for result in results
    )
    memory_type_total = sum(
        result.memory_type_total
        for result in results
    )

    summary = {
        "case_count": case_count,
        "passed_cases": sum(result.passed for result in results),
        "case_pass_rate": _rate(
            sum(result.passed for result in results),
            case_count,
        ),
        "final_state_accuracy": _rate(
            sum(result.passed for result in results),
            case_count,
        ),
        "expected_change_applied_rate": _rate(
            expected_change_matches,
            expected_change_total,
        ),
        "memory_type_accuracy": _rate(
            memory_type_matches,
            memory_type_total,
        ),
        "unsafe_final_mutation_rate": _rate(
            sum(result.unsafe_final_mutation for result in results),
            case_count,
        ),
        "incorrect_deletion_rate": _rate(
            sum(result.incorrect_deletion for result in results),
            case_count,
        ),
        "incorrect_update_rate": _rate(
            sum(result.incorrect_update for result in results),
            case_count,
        ),
        "unexpected_creation_rate": _rate(
            sum(result.unexpected_creation for result in results),
            case_count,
        ),
        "duplicate_active_memory_rate": _rate(
            sum(
                result.duplicate_active_memory
                for result in results
            ),
            case_count,
        ),
        "unnecessary_update_rate": _rate(
            sum(result.unnecessary_update for result in results),
            case_count,
        ),
    }

    return {
        "schema_version": "1.0",
        "dataset_schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "model": model_name,
        "summary": summary,
        "cases": [
            asdict(result)
            for result in results
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]

    print("\nMemory pipeline evaluation")
    print("=" * 40)
    print(f"Model: {report['model']}")
    print(
        f"Cases passed: "
        f"{summary['passed_cases']}/{summary['case_count']}"
    )

    for metric in (
        "case_pass_rate",
        "final_state_accuracy",
        "expected_change_applied_rate",
        "memory_type_accuracy",
        "unsafe_final_mutation_rate",
        "incorrect_deletion_rate",
        "incorrect_update_rate",
        "unexpected_creation_rate",
        "duplicate_active_memory_rate",
        "unnecessary_update_rate",
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
                "  final active: "
                f"{case['final_active_memories']}"
            )
            print(
                "  deleted ids: "
                f"{case['actual_deleted_ids']}"
            )


async def evaluate(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )
    settings = get_settings()
    llm_client = get_memory_llm()
    prompt_repository = get_prompt_repository()
    planner = LLMMemoryPlanner(
        llm_client=llm_client,
        prompt_repository=prompt_repository,
    )
    validator = LLMMemoryMutationValidator(
        llm_client=llm_client,
        prompt_repository=prompt_repository,
    )

    results: list[PipelineCaseResult] = []

    for position, case in enumerate(dataset["cases"], start=1):
        print(
            f"[{position}/{len(dataset['cases'])}] "
            f"{case['id']}"
        )
        results.append(
            await _evaluate_case(
                planner=planner,
                validator=validator,
                case=case,
                confidence_threshold=(
                    settings.memory_confidence_threshold
                ),
                retrieval_limit=settings.memory_retrieval_limit,
            )
        )

    report = _build_report(
        dataset=dataset,
        results=results,
        model_name=getattr(
            llm_client,
            "model",
            type(llm_client).__name__,
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_report(report)
    print(f"\nReport written to: {report_path}")
    return report


def _parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate planner, mutation validator, and manager against "
            "expected final memory state."
        )
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
