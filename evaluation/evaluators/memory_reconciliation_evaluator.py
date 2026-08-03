from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from pydantic import TypeAdapter

from memory_engine.memory.extractor import MemoryFact
from memory_engine.memory.reconciler import (
    CreateDecision,
    InvalidateDecision,
    MemoryReconciliationDecision,
    MemoryReconciliationRequest,
    MemoryReconciler,
    UpdateDecision,
)
from memory_engine.memory.service import Memory
from memory_engine.models import MemoryType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_reconciliation.json"
)
DEFAULT_HOLDOUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_reconciliation_holdout.json"
)
DEFAULT_HOLDOUT_V2_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_reconciliation_holdout_v2.json"
)
DEFAULT_HOLDOUT_V3_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_reconciliation_holdout_v3.json"
)
DEFAULT_HOLDOUT_V4_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_reconciliation_holdout_v4.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_reconciliation_evaluation.json"
)
FACT_ADAPTER = TypeAdapter(MemoryFact)


@dataclass(frozen=True)
class ReconciliationCaseResult:
    case_id: str
    passed: bool
    schema_valid: bool
    fact_count: int
    decision_count: int
    action_matches: int
    target_matches: int
    target_total: int
    content_matches: int
    content_total: int
    fact_isolation_matches: int
    false_create_count: int
    false_destructive_mutation_count: int
    duplicate_create: bool
    batch_constraints_correct: bool
    latency_ms: float
    expected_decisions: list[dict[str, Any]]
    actual_decisions: list[dict[str, Any]]
    error: str | None = None


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate batched semantic memory reconciliation.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--rescore-from",
        type=Path,
        help="Rescore recorded decisions without another model call.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this development case; may be repeated.",
    )
    parser.add_argument(
        "--model",
        help="Explicit OpenAI model used for this evaluation.",
    )
    return parser.parse_args(args)


def _build_requests(
    case: dict[str, Any],
) -> tuple[MemoryReconciliationRequest, ...]:
    requests: list[MemoryReconciliationRequest] = []
    for fact_case in case["facts"]:
        fact = FACT_ADAPTER.validate_python(fact_case["fact"])
        candidates = tuple(
            Memory(
                id=record["id"],
                content=record["content"],
                memory_type=MemoryType(record["memory_type"]),
                confidence=float(record["confidence"]),
            )
            for record in fact_case.get("candidate_memories", [])
        )
        requests.append(
            MemoryReconciliationRequest(
                fact=fact,
                candidate_memories=candidates,
            )
        )
    return tuple(requests)


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


def _tokens(text: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
    ]


def _contains_concept(content: str, concept: str) -> bool:
    content_tokens = _tokens(content)
    concept_tokens = _tokens(concept)
    if not concept_tokens:
        return True
    position = 0
    for token in content_tokens:
        if token == concept_tokens[position]:
            position += 1
            if position == len(concept_tokens):
                return True
    return False


def _contains_contiguous_concept(content: str, concept: str) -> bool:
    content_tokens = _tokens(content)
    concept_tokens = _tokens(concept)
    if not concept_tokens:
        return True
    width = len(concept_tokens)
    return any(
        content_tokens[index : index + width] == concept_tokens
        for index in range(len(content_tokens) - width + 1)
    )


def _content_passes(content: str, assertions: dict[str, Any]) -> bool:
    required = all(
        _contains_concept(content, concept)
        for concept in assertions.get("must_include", [])
    )
    alternatives = assertions.get("must_include_any", [])
    alternative = not alternatives or any(
        all(_contains_concept(content, concept) for concept in option)
        for option in alternatives
    )
    forbidden = all(
        not _contains_contiguous_concept(content, concept)
        for concept in assertions.get("must_not_include", [])
    )
    return required and alternative and forbidden


def _action(decision: MemoryReconciliationDecision) -> str:
    return decision.action


def _target_id(
    decision: MemoryReconciliationDecision,
    request: MemoryReconciliationRequest,
) -> str | None:
    if not isinstance(decision, (UpdateDecision, InvalidateDecision)):
        return None
    if decision.memory_index >= len(request.candidate_memories):
        return None
    return request.candidate_memories[decision.memory_index].id


def _serialize_decision(
    decision: MemoryReconciliationDecision,
    request: MemoryReconciliationRequest,
) -> dict[str, Any]:
    serialized = decision.model_dump(mode="json")
    target_id = _target_id(decision, request)
    if target_id is not None:
        serialized["target_memory_id"] = target_id
    return serialized


def _accepted_actions(expected: dict[str, Any]) -> set[str]:
    return set(
        expected.get("accepted_actions", [expected["action"]])
    )


def _batch_constraints_correct(
    case: dict[str, Any],
    decisions: tuple[MemoryReconciliationDecision, ...],
    requests: tuple[MemoryReconciliationRequest, ...],
) -> tuple[bool, bool]:
    constraints = case.get("batch_constraints", {}).get(
        "create_counts",
        [],
    )
    correct = True
    duplicate = False
    for constraint in constraints:
        count = sum(
            isinstance(decisions[index], CreateDecision)
            for index in constraint["fact_indices"]
        )
        minimum = int(constraint.get("minimum", 0))
        maximum = int(constraint["maximum"])
        correct = correct and minimum <= count <= maximum
        duplicate = duplicate or count > maximum
    for constraint in case.get("batch_constraints", {}).get(
        "mutation_counts",
        [],
    ):
        allowed_actions = set(
            constraint.get("actions", ["update", "invalidate"])
        )
        accepted_targets = set(constraint.get("target_memory_ids", []))
        matching = 0
        for index in constraint["fact_indices"]:
            decision = decisions[index]
            if _action(decision) not in allowed_actions:
                continue
            if accepted_targets:
                if (
                    _target_id(decision, requests[index])
                    not in accepted_targets
                ):
                    continue
            matching += 1
        minimum = int(constraint.get("minimum", 0))
        maximum = int(constraint["maximum"])
        correct = correct and minimum <= matching <= maximum
    return correct, duplicate


async def _score_case(
    case: dict[str, Any],
    reconciler: MemoryReconciler,
) -> ReconciliationCaseResult:
    requests = _build_requests(case)
    expected = [item["expected"] for item in case["facts"]]
    started = perf_counter()
    try:
        decisions = await reconciler.reconcile_many(requests)
    except Exception as error:
        return ReconciliationCaseResult(
            case_id=case["id"],
            passed=False,
            schema_valid=False,
            fact_count=len(requests),
            decision_count=0,
            action_matches=0,
            target_matches=0,
            target_total=sum(
                "target_memory_ids" in item for item in expected
            ),
            content_matches=0,
            content_total=sum(
                "content_assertions" in item for item in expected
            ),
            fact_isolation_matches=0,
            false_create_count=0,
            false_destructive_mutation_count=0,
            duplicate_create=False,
            batch_constraints_correct=False,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            expected_decisions=expected,
            actual_decisions=[],
            error=f"{type(error).__name__}: {error}",
        )

    latency_ms = round((perf_counter() - started) * 1000, 3)
    if len(decisions) != len(requests):
        return ReconciliationCaseResult(
            case_id=case["id"],
            passed=False,
            schema_valid=False,
            fact_count=len(requests),
            decision_count=len(decisions),
            action_matches=0,
            target_matches=0,
            target_total=sum(
                "target_memory_ids" in item for item in expected
            ),
            content_matches=0,
            content_total=sum(
                "content_assertions" in item for item in expected
            ),
            fact_isolation_matches=0,
            false_create_count=0,
            false_destructive_mutation_count=0,
            duplicate_create=False,
            batch_constraints_correct=False,
            latency_ms=latency_ms,
            expected_decisions=expected,
            actual_decisions=[],
            error="Decision count did not match fact count.",
        )

    action_matches = 0
    target_matches = 0
    target_total = 0
    content_matches = 0
    content_total = 0
    fact_isolation_matches = 0
    false_create_count = 0
    false_destructive_count = 0
    actual = []

    for request, decision, expected_decision in zip(
        requests,
        decisions,
        expected,
        strict=True,
    ):
        actual_action = _action(decision)
        accepted = _accepted_actions(expected_decision)
        action_correct = actual_action in accepted
        action_matches += action_correct
        target_correct = True
        target_is_optional = "noop" in accepted
        if "target_memory_ids" in expected_decision and not (
            target_is_optional and actual_action == "noop"
        ):
            target_total += 1
            target_correct = _target_id(decision, request) in set(
                expected_decision["target_memory_ids"]
            )
            target_matches += target_correct
        content_correct = True
        if "content_assertions" in expected_decision and not (
            target_is_optional and actual_action == "noop"
        ):
            content_total += 1
            content_correct = (
                isinstance(decision, UpdateDecision)
                and _content_passes(
                    decision.content,
                    expected_decision["content_assertions"],
                )
            )
            content_matches += content_correct

        false_create = (
            isinstance(decision, CreateDecision)
            and "create" not in accepted
        )
        false_destructive = (
            isinstance(decision, (UpdateDecision, InvalidateDecision))
            and (
                actual_action not in accepted
                or not target_correct
            )
        )
        false_create_count += false_create
        false_destructive_count += false_destructive
        isolated = (
            action_correct
            and target_correct
            and not false_create
            and not false_destructive
        )
        fact_isolation_matches += isolated
        actual.append(_serialize_decision(decision, request))

    constraints_correct, duplicate_create = _batch_constraints_correct(
        case,
        decisions,
        requests,
    )
    passed = (
        action_matches == len(requests)
        and target_matches == target_total
        and content_matches == content_total
        and fact_isolation_matches == len(requests)
        and false_create_count == 0
        and false_destructive_count == 0
        and constraints_correct
    )
    return ReconciliationCaseResult(
        case_id=case["id"],
        passed=passed,
        schema_valid=True,
        fact_count=len(requests),
        decision_count=len(decisions),
        action_matches=action_matches,
        target_matches=target_matches,
        target_total=target_total,
        content_matches=content_matches,
        content_total=content_total,
        fact_isolation_matches=fact_isolation_matches,
        false_create_count=false_create_count,
        false_destructive_mutation_count=false_destructive_count,
        duplicate_create=duplicate_create,
        batch_constraints_correct=constraints_correct,
        latency_ms=latency_ms,
        expected_decisions=expected,
        actual_decisions=actual,
    )


def _rate(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else round(numerator / denominator, 4)


def _build_report(
    dataset: dict[str, Any],
    results: list[ReconciliationCaseResult],
    *,
    model: str | None,
) -> dict[str, Any]:
    case_count = len(results)
    fact_count = sum(item.fact_count for item in results)
    target_total = sum(item.target_total for item in results)
    content_total = sum(item.content_total for item in results)
    summary = {
        "case_pass_rate": _rate(
            sum(item.passed for item in results), case_count
        ),
        "schema_valid_response_rate": _rate(
            sum(item.schema_valid for item in results), case_count
        ),
        "batch_complete_rate": _rate(
            sum(
                item.schema_valid
                and item.decision_count == item.fact_count
                for item in results
            ),
            case_count,
        ),
        "operation_accuracy": _rate(
            sum(item.action_matches for item in results), fact_count
        ),
        "target_index_accuracy": _rate(
            sum(item.target_matches for item in results), target_total
        ),
        "update_content_faithfulness_rate": _rate(
            sum(item.content_matches for item in results), content_total
        ),
        "fact_isolation_rate": _rate(
            sum(item.fact_isolation_matches for item in results), fact_count
        ),
        "false_create_rate": _rate(
            sum(item.false_create_count for item in results), fact_count
        ),
        "false_destructive_mutation_rate": _rate(
            sum(
                item.false_destructive_mutation_count for item in results
            ),
            fact_count,
        ),
        "duplicate_create_case_rate": _rate(
            sum(item.duplicate_create for item in results), case_count
        ),
        "batch_constraint_accuracy": _rate(
            sum(item.batch_constraints_correct for item in results),
            case_count,
        ),
        "average_latency_ms": round(
            sum(item.latency_ms for item in results) / case_count,
            3,
        )
        if case_count
        else 0.0,
        "average_facts_per_batch": round(fact_count / case_count, 3)
        if case_count
        else 0.0,
    }
    checks: dict[str, bool] = {}
    for metric, boundary in dataset["evaluation_contract"][
        "acceptance_gates"
    ].items():
        value = summary[metric]
        checks[metric] = (
            value >= boundary["minimum"]
            if "minimum" in boundary
            else value <= boundary["maximum"]
        )
    return {
        "schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "dataset_description": dataset["description"],
        "model": model,
        "summary": summary,
        "acceptance_gates": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "cases": [asdict(item) for item in results],
    }


async def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    *,
    reconciler: MemoryReconciler,
    model: str | None = None,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if case_ids is not None:
        known = {case["id"] for case in cases}
        unknown = case_ids - known
        if unknown:
            raise ValueError(
                "Unknown memory reconciliation case IDs: "
                + ", ".join(sorted(unknown))
            )
        cases = [case for case in cases if case["id"] in case_ids]
    results = [await _score_case(case, reconciler) for case in cases]
    report = _build_report(dataset, results, model=model)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def rescore_report(
    dataset_path: Path,
    source_report_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    cases_by_id = {case["id"]: case for case in dataset["cases"]}
    results: list[ReconciliationCaseResult] = []
    for stored in source["cases"]:
        case = cases_by_id[stored["case_id"]]
        content_total = 0
        content_matches = 0
        for fact_case, actual in zip(
            case["facts"], stored["actual_decisions"], strict=True
        ):
            expected = fact_case["expected"]
            if "content_assertions" not in expected:
                continue
            accepted = _accepted_actions(expected)
            if "noop" in accepted and actual["action"] == "noop":
                continue
            content_total += 1
            content_matches += int(
                actual["action"] == "update"
                and _content_passes(
                    actual["content"], expected["content_assertions"]
                )
            )
        updated = dict(stored)
        updated["content_total"] = content_total
        updated["content_matches"] = content_matches
        updated["passed"] = (
            updated["schema_valid"]
            and updated["action_matches"] == updated["fact_count"]
            and updated["target_matches"] == updated["target_total"]
            and content_matches == content_total
            and updated["fact_isolation_matches"] == updated["fact_count"]
            and updated["false_create_count"] == 0
            and updated["false_destructive_mutation_count"] == 0
            and updated["batch_constraints_correct"]
        )
        results.append(ReconciliationCaseResult(**updated))
    report = _build_report(dataset, results, model=source.get("model"))
    report["rescored_from"] = str(source_report_path)
    if "usage" in source:
        report["usage"] = source["usage"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(args: list[str] | None = None) -> int:
    from memory_engine.config import get_settings
    from memory_engine.dependencies import get_memory_llm, get_prompt_repository
    from memory_engine.memory.reconciler import LLMMemoryReconciler
    from evaluation.usage import attach_usage_and_rewrite
    from evaluation.clients import build_openai_evaluation_client

    parsed = _parse_args(args)
    if parsed.rescore_from:
        report = rescore_report(
            parsed.dataset,
            parsed.rescore_from,
            parsed.report,
        )
        print(json.dumps(report["summary"], indent=2))
        print(
            "Acceptance gates: "
            + ("PASSED" if report["acceptance_gates"]["passed"] else "FAILED")
        )
        print(f"Report: {parsed.report}")
        return 0 if report["acceptance_gates"]["passed"] else 1
    settings = get_settings()
    client = (
        build_openai_evaluation_client(parsed.model)
        if parsed.model
        else get_memory_llm()
    )
    report = asyncio.run(
        run_evaluation(
            parsed.dataset,
            parsed.report,
            reconciler=LLMMemoryReconciler(
                llm_client=client,
                prompt_repository=get_prompt_repository(),
            ),
            model=(
                parsed.model
                or (
                    settings.semantic_model
                    if settings.memory_use_semantic_model
                    else settings.memory_model
                )
            ),
            case_ids=set(parsed.case_ids) if parsed.case_ids else None,
        )
    )
    attach_usage_and_rewrite(report, parsed.report, client)
    print(json.dumps(report["summary"], indent=2))
    print(
        "Acceptance gates: "
        + ("PASSED" if report["acceptance_gates"]["passed"] else "FAILED")
    )
    print(f"Report: {parsed.report}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
