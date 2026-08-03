from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from memory_engine.dependencies import get_memory_llm, get_prompt_repository
from memory_engine.memory.planner import (
    DeleteMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import Memory
from memory_engine.memory.validator import (
    LLMMemoryMutationValidator,
    MemoryMutationValidation,
)
from memory_engine.models import MemoryType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_mutation_validation.json"
)


@dataclass(frozen=True)
class ValidationCaseResult:
    case_id: str
    expected_approved: bool
    actual_approved: bool
    passed: bool
    schema_valid: bool
    reason: str
    latency_ms: float


def _build_operation(case: dict[str, Any]):
    common = {
        "memory_index": 0,
        "confidence": 1.0,
        "explanation": "Proposed by the planner under evaluation.",
    }
    if case["operation"] == "update":
        return UpdateMemoryOperation(
            operation="update",
            content=case["proposed_content"],
            **common,
        )
    return DeleteMemoryOperation(
        operation="delete",
        **common,
    )


async def _evaluate_case(
    validator: LLMMemoryMutationValidator,
    case: dict[str, Any],
) -> ValidationCaseResult:
    target = case["target_memory"]
    operation = _build_operation(case)
    target_memory = Memory(
        id="evaluation-target",
        content=target["content"],
        memory_type=MemoryType(target["memory_type"]),
        confidence=1.0,
    )
    messages = validator._build_messages(
        user_message=case["user_message"],
        operation=operation,
        target_memory=target_memory,
    )

    started = perf_counter()
    try:
        raw_response = await validator.llm_client.generate(
            messages=messages,
            response_schema=(
                MemoryMutationValidation.model_json_schema()
            ),
        )
        result = MemoryMutationValidation.model_validate_json(
            raw_response
        )
        schema_valid = True
    except (RuntimeError, ValidationError) as error:
        result = MemoryMutationValidation(
            approved=False,
            reason=f"Invalid validator response: {error}",
        )
        schema_valid = False
    latency_ms = round((perf_counter() - started) * 1000, 3)

    expected = case["expected"]["approved"]
    return ValidationCaseResult(
        case_id=case["id"],
        expected_approved=expected,
        actual_approved=result.approved,
        passed=schema_valid and result.approved == expected,
        schema_valid=schema_valid,
        reason=result.reason,
        latency_ms=latency_ms,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _build_report(
    dataset: dict[str, Any],
    results: list[ValidationCaseResult],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    unsafe = [
        result
        for result in results
        if not result.expected_approved
    ]
    safe = [
        result
        for result in results
        if result.expected_approved
    ]
    summary = {
        "case_count": len(results),
        "passed_cases": sum(result.passed for result in results),
        "schema_valid_response_rate": _rate(
            sum(result.schema_valid for result in results),
            len(results),
        ),
        "decision_accuracy": _rate(
            sum(result.passed for result in results),
            len(results),
        ),
        "unsafe_approval_rate": _rate(
            sum(result.actual_approved for result in unsafe),
            len(unsafe),
        ),
        "safe_approval_rate": _rate(
            sum(result.actual_approved for result in safe),
            len(safe),
        ),
        "safe_rejection_rate": _rate(
            sum(not result.actual_approved for result in safe),
            len(safe),
        ),
        "average_latency_ms": (
            round(
                sum(result.latency_ms for result in results) / len(results),
                3,
            )
            if results
            else 0.0
        ),
    }
    checks: dict[str, bool] = {}
    for metric, boundary in dataset["evaluation_contract"].get(
        "acceptance_gates",
        {},
    ).items():
        value = summary[metric]
        checks[metric] = (
            value >= boundary["minimum"]
            if "minimum" in boundary
            else value <= boundary["maximum"]
        )
    return {
        "schema_version": "1.0",
        "dataset_schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "model": model,
        "summary": summary,
        "acceptance_gates": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "cases": [asdict(result) for result in results],
    }


async def evaluate(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    *,
    validator: LLMMemoryMutationValidator | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    from memory_engine.config import get_settings

    settings = get_settings()
    configured_validator = validator or LLMMemoryMutationValidator(
        llm_client=get_memory_llm(),
        prompt_repository=get_prompt_repository(),
    )
    results = [
        await _evaluate_case(configured_validator, case)
        for case in dataset["cases"]
    ]
    report = _build_report(
        dataset,
        results,
        model=(
            model
            or (
                settings.semantic_model
                if settings.memory_use_semantic_model
                else settings.memory_model
            )
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the destructive memory-mutation validator.",
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
    parser.add_argument(
        "--model",
        help="Explicit OpenAI model used for this evaluation.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    from evaluation.clients import build_openai_evaluation_client
    from evaluation.usage import attach_usage_and_rewrite

    parsed = _parse_args(args)
    client = (
        build_openai_evaluation_client(parsed.model)
        if parsed.model
        else get_memory_llm()
    )
    report = asyncio.run(
        evaluate(
            parsed.dataset,
            parsed.report,
            validator=LLMMemoryMutationValidator(
                llm_client=client,
                prompt_repository=get_prompt_repository(),
            ),
            model=parsed.model,
        )
    )
    attach_usage_and_rewrite(
        report,
        parsed.report,
        client,
    )
    print(json.dumps(report["summary"], indent=2))
    print(
        "Acceptance gates: "
        + ("PASSED" if report["acceptance_gates"]["passed"] else "FAILED")
    )
    print(f"Report written to: {parsed.report}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
