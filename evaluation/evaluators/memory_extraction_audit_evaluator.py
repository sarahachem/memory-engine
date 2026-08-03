from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from memory_engine.config import get_settings
from memory_engine.dependencies import get_memory_llm, get_prompt_repository
from memory_engine.memory.extraction_auditor import (
    LLMMemoryExtractionAuditor,
    MemoryExtractionAuditor,
)
from memory_engine.memory.extractor import LLMMemoryFactExtractor, MemoryFactExtractor
from evaluation.evaluators.memory_extraction_evaluator import (
    DEFAULT_DATASET_PATH,
    _build_report,
    _load_dataset,
    _score_case,
    _serialize_facts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_extraction_audit_evaluation.json"
)


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extraction alone with one bounded audit pass."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(args)


def _safe_rate(value: int, total: int) -> float:
    return 0.0 if total == 0 else round(value / total, 4)


async def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    *,
    extractor: MemoryFactExtractor | None = None,
    auditor: MemoryExtractionAuditor | None = None,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    settings = get_settings()
    configured_extractor = extractor or LLMMemoryFactExtractor(
        llm_client=get_memory_llm(),
        prompt_repository=get_prompt_repository(),
    )
    configured_auditor = auditor or LLMMemoryExtractionAuditor(
        llm_client=get_memory_llm(),
        prompt_repository=get_prompt_repository(),
    )
    baseline_results = []
    audited_results = []
    case_details = []

    for case in dataset["cases"]:
        try:
            candidates = await configured_extractor.extract(
                case["user_message"]
            )
            candidate_data = _serialize_facts(candidates)
            baseline = _score_case(
                case,
                candidate_data,
                schema_valid=True,
                confidence_threshold=settings.memory_confidence_threshold,
            )
        except Exception as error:
            baseline = _score_case(
                case,
                [],
                schema_valid=False,
                confidence_threshold=settings.memory_confidence_threshold,
                error=f"{type(error).__name__}: {error}",
            )
            baseline_results.append(baseline)
            audited_results.append(baseline)
            case_details.append(
                {
                    "case_id": case["id"],
                    "baseline_passed": False,
                    "audit_approved": None,
                    "audit_issues": [],
                    "audited_passed": False,
                    "error": baseline.error,
                }
            )
            continue

        baseline_results.append(baseline)
        try:
            audit = await configured_auditor.audit(
                user_message=case["user_message"],
                candidate_facts=candidates,
            )
            final_data = _serialize_facts(audit.final_facts)
            audited = _score_case(
                case,
                final_data,
                schema_valid=True,
                confidence_threshold=settings.memory_confidence_threshold,
            )
            detail_error = None
        except Exception as error:
            audit = None
            audited = _score_case(
                case,
                [],
                schema_valid=False,
                confidence_threshold=settings.memory_confidence_threshold,
                error=f"{type(error).__name__}: {error}",
            )
            detail_error = audited.error
        audited_results.append(audited)
        case_details.append(
            {
                "case_id": case["id"],
                "baseline_passed": baseline.passed,
                "audit_approved": None if audit is None else audit.approved,
                "audit_issues": (
                    []
                    if audit is None
                    else [
                        issue.model_dump(mode="json")
                        for issue in audit.issues
                    ]
                ),
                "audited_passed": audited.passed,
                "candidate_facts": baseline.actual_facts,
                "final_facts": audited.actual_facts,
                "error": detail_error,
            }
        )

    model = None if extractor is not None else settings.memory_model
    baseline_report = _build_report(
        dataset,
        baseline_results,
        model=model,
        confidence_threshold=settings.memory_confidence_threshold,
    )
    audited_report = _build_report(
        dataset,
        audited_results,
        model=model,
        confidence_threshold=settings.memory_confidence_threshold,
    )
    baseline_passes = sum(result.passed for result in baseline_results)
    baseline_failures = len(baseline_results) - baseline_passes
    regressions = sum(
        before.passed and not after.passed
        for before, after in zip(baseline_results, audited_results)
    )
    recoveries = sum(
        not before.passed and after.passed
        for before, after in zip(baseline_results, audited_results)
    )
    audit_failures = sum(
        detail["audit_approved"] is None and detail["error"] is not None
        for detail in case_details
    )
    corrections = sum(
        detail["audit_approved"] is False for detail in case_details
    )
    report = {
        "schema_version": dataset["schema_version"],
        "evaluation_target": "MemoryFactExtractor+MemoryExtractionAuditor",
        "dataset_description": dataset["description"],
        "model": model,
        "baseline_summary": baseline_report["summary"],
        "audited_summary": audited_report["summary"],
        "audit_summary": {
            "correction_rate": _safe_rate(corrections, len(case_details)),
            "recovery_rate": _safe_rate(recoveries, baseline_failures),
            "regression_rate": _safe_rate(regressions, baseline_passes),
            "audit_failure_rate": _safe_rate(
                audit_failures, len(case_details)
            ),
        },
        "acceptance_gates": audited_report["acceptance_gates"],
        "cases": case_details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(args: list[str] | None = None) -> int:
    parsed = _parse_args(args)
    report = asyncio.run(run_evaluation(parsed.dataset, parsed.report))
    print(json.dumps(report["audited_summary"], indent=2))
    print(json.dumps(report["audit_summary"], indent=2))
    print(f"Report: {parsed.report.resolve()}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
