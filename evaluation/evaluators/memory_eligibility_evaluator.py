from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from memory_engine.config import get_settings
from memory_engine.dependencies import get_memory_llm, get_prompt_repository
from memory_engine.memory.eligibility import LLMMemoryEligibilityGate, MemoryEligibilityGate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation/datasets/memory_eligibility.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation/reports/memory_eligibility_evaluation.json"


@dataclass(frozen=True)
class EligibilityCaseResult:
    case_id: str
    expected_eligible: bool
    actual_eligible: bool | None
    passed: bool
    schema_valid: bool
    reason: str | None
    confidence: float | None
    latency_ms: float
    error: str | None = None


def _rate(value: int, total: int, empty: float = 1.0) -> float:
    return empty if total == 0 else round(value / total, 4)


async def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    *,
    gate: MemoryEligibilityGate | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    settings = get_settings()
    configured_gate = gate or LLMMemoryEligibilityGate(
        get_memory_llm(), get_prompt_repository()
    )
    results = []
    for case in dataset["cases"]:
        started = perf_counter()
        try:
            assessment = await configured_gate.assess(case["message"])
            actual = assessment.eligible
            results.append(
                EligibilityCaseResult(
                    case_id=case["id"],
                    expected_eligible=case["expected_eligible"],
                    actual_eligible=actual,
                    passed=actual is case["expected_eligible"],
                    schema_valid=True,
                    reason=assessment.reason.value,
                    confidence=assessment.confidence,
                    latency_ms=round((perf_counter() - started) * 1000, 3),
                )
            )
        except Exception as error:
            results.append(
                EligibilityCaseResult(
                    case_id=case["id"],
                    expected_eligible=case["expected_eligible"],
                    actual_eligible=None,
                    passed=False,
                    schema_valid=False,
                    reason=None,
                    confidence=None,
                    latency_ms=round((perf_counter() - started) * 1000, 3),
                    error=f"{type(error).__name__}: {error}",
                )
            )
    positives = [item for item in results if item.expected_eligible]
    negatives = [item for item in results if not item.expected_eligible]
    false_negatives = sum(item.actual_eligible is False for item in positives)
    summary = {
        "case_pass_rate": _rate(sum(item.passed for item in results), len(results)),
        "schema_valid_response_rate": _rate(sum(item.schema_valid for item in results), len(results)),
        "eligible_recall": _rate(sum(item.actual_eligible is True for item in positives), len(positives)),
        "ineligible_accuracy": _rate(sum(item.actual_eligible is False for item in negatives), len(negatives)),
        "false_negative_rate": _rate(false_negatives, len(positives), 0.0),
        "average_latency_ms": round(sum(item.latency_ms for item in results) / len(results), 3),
    }
    checks = {}
    for metric, boundary in dataset["evaluation_contract"]["acceptance_gates"].items():
        checks[metric] = (
            summary[metric] >= boundary["minimum"]
            if "minimum" in boundary
            else summary[metric] <= boundary["maximum"]
        )
    report = {
        "schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "dataset_description": dataset["description"],
        "model": None if gate is not None else settings.memory_model,
        "summary": summary,
        "acceptance_gates": {"passed": all(checks.values()), "checks": checks},
        "cases": [asdict(item) for item in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate memory eligibility.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parsed = parser.parse_args(args)
    report = asyncio.run(run_evaluation(parsed.dataset, parsed.report))
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {parsed.report.resolve()}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
