from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from memory_engine.config import get_settings
from memory_engine.dependencies import get_memory_retriever
from memory_engine.memory.retriever import MemoryRetriever
from memory_engine.memory.service import Memory, MemoryStatus
from memory_engine.models import MemoryType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "datasets" / "memory_retrieval.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_retrieval_evaluation.json"
)


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    passed: bool
    limit: int
    expected_relevant_ids: list[str]
    critical_relevant_ids: list[str]
    returned_ids: list[str]
    returned_scores: list[float]
    relevant_returned_count: int
    irrelevant_returned_count: int
    critical_miss_count: int
    inactive_leak_count: int
    reciprocal_rank: float


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic memory retrieval.",
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


def _build_memories(case: dict[str, Any]) -> tuple[Memory, ...]:
    return tuple(
        Memory(
            id=item["id"],
            content=item["content"],
            memory_type=MemoryType(item["memory_type"]),
            confidence=float(item["confidence"]),
            status=MemoryStatus(item.get("status", "active")),
        )
        for item in case["candidate_memories"]
    )


async def _score_case(
    case: dict[str, Any],
    retriever: MemoryRetriever,
) -> RetrievalCaseResult:
    candidates = _build_memories(case)
    retrieved = await retriever.retrieve(
        query=case["query"],
        candidate_memories=candidates,
        limit=int(case["limit"]),
    )
    returned_ids = [result.memory.id for result in retrieved]
    expected = set(case["expected_relevant_ids"])
    critical = set(case["critical_relevant_ids"])
    inactive = {
        memory.id
        for memory in candidates
        if memory.status is not MemoryStatus.ACTIVE
    }
    relevant_positions = [
        index
        for index, memory_id in enumerate(returned_ids, start=1)
        if memory_id in expected
    ]
    relevant_count = len(expected.intersection(returned_ids))
    irrelevant_count = sum(
        memory_id not in expected for memory_id in returned_ids
    )
    critical_misses = len(critical.difference(returned_ids))
    inactive_leaks = len(inactive.intersection(returned_ids))
    required_relevant = min(int(case["limit"]), len(expected))
    passed = (
        len(returned_ids) <= int(case["limit"])
        and relevant_count == required_relevant
        and irrelevant_count == 0
        and critical_misses == 0
        and inactive_leaks == 0
    )

    return RetrievalCaseResult(
        case_id=case["id"],
        passed=passed,
        limit=int(case["limit"]),
        expected_relevant_ids=list(case["expected_relevant_ids"]),
        critical_relevant_ids=list(case["critical_relevant_ids"]),
        returned_ids=returned_ids,
        returned_scores=[round(result.score, 6) for result in retrieved],
        relevant_returned_count=relevant_count,
        irrelevant_returned_count=irrelevant_count,
        critical_miss_count=critical_misses,
        inactive_leak_count=inactive_leaks,
        reciprocal_rank=(
            0.0
            if not relevant_positions
            else round(1.0 / relevant_positions[0], 4)
        ),
    )


def _rate(
    numerator: int | float,
    denominator: int,
    *,
    empty_value: float = 0.0,
) -> float:
    if denominator == 0:
        return empty_value
    return round(numerator / denominator, 4)


def _evaluate_gates(
    summary: dict[str, float],
    gates: dict[str, dict[str, float]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for metric, boundary in gates.items():
        if "minimum" in boundary:
            checks[metric] = summary[metric] >= boundary["minimum"]
        elif "maximum" in boundary:
            checks[metric] = summary[metric] <= boundary["maximum"]
        else:
            raise ValueError(f"Gate for {metric} has no boundary.")
    return {"passed": all(checks.values()), "checks": checks}


def _build_report(
    dataset: dict[str, Any],
    results: list[RetrievalCaseResult],
    *,
    model: str | None,
    technical_floor: float | None,
) -> dict[str, Any]:
    relevant_cases = [
        result for result in results if result.expected_relevant_ids
    ]
    total_expected = sum(
        min(result.limit, len(result.expected_relevant_ids))
        for result in results
    )
    total_returned = sum(len(result.returned_ids) for result in results)
    total_relevant_returned = sum(
        result.relevant_returned_count for result in results
    )
    total_critical = sum(
        len(result.critical_relevant_ids) for result in results
    )
    inactive_cases = [
        case
        for case in dataset["cases"]
        if any(
            memory.get("status", "active") != "active"
            for memory in case["candidate_memories"]
        )
    ]
    result_by_id = {result.case_id: result for result in results}
    inactive_case_leaks = sum(
        result_by_id[case["id"]].inactive_leak_count > 0
        for case in inactive_cases
    )
    summary = {
        "case_pass_rate": _rate(
            sum(result.passed for result in results),
            len(results),
            empty_value=1.0,
        ),
        "recall_at_k": _rate(
            total_relevant_returned,
            total_expected,
            empty_value=1.0,
        ),
        "precision_at_k": _rate(
            total_relevant_returned,
            total_returned,
            empty_value=1.0,
        ),
        "mean_reciprocal_rank": _rate(
            sum(result.reciprocal_rank for result in relevant_cases),
            len(relevant_cases),
            empty_value=1.0,
        ),
        "critical_memory_miss_rate": _rate(
            sum(result.critical_miss_count for result in results),
            total_critical,
        ),
        "irrelevant_memory_inclusion_rate": _rate(
            sum(result.irrelevant_returned_count for result in results),
            total_returned,
        ),
        "inactive_memory_leakage_rate": _rate(
            inactive_case_leaks,
            len(inactive_cases),
        ),
    }
    gates = dataset["evaluation_contract"].get(
        "acceptance_gates", {}
    )
    return {
        "schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "dataset_description": dataset["description"],
        "model": model,
        "technical_floor": technical_floor,
        "summary": summary,
        "acceptance_gates": _evaluate_gates(summary, gates),
        "cases": [asdict(result) for result in results],
    }


async def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    *,
    retriever: MemoryRetriever | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    configured_retriever = retriever or get_memory_retriever()
    results = [
        await _score_case(case, configured_retriever)
        for case in dataset["cases"]
    ]
    settings = get_settings()
    report = _build_report(
        dataset,
        results,
        model=None if retriever is not None else settings.embedding_model,
        technical_floor=getattr(
            configured_retriever, "technical_floor", None
        ),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main(args: list[str] | None = None) -> int:
    parsed = _parse_args(args)
    report = asyncio.run(
        run_evaluation(parsed.dataset, parsed.report)
    )
    print(json.dumps(report["summary"], indent=2))
    print(
        "Acceptance gates: "
        + (
            "PASSED"
            if report["acceptance_gates"]["passed"]
            else "FAILED"
        )
    )
    print(f"Report: {parsed.report}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
