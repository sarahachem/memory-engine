from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from memory_engine.memory.reranker import MemoryReranker
from memory_engine.memory.retriever import RetrievedMemory
from memory_engine.memory.service import Memory, MemoryStatus
from memory_engine.models import Intent, MemoryType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "datasets" / "context_memory.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "reports" / "context_memory_evaluation.json"
)


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final semantic context-memory reranking.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--provider",
        choices=("cross_encoder", "ollama", "openai"),
        default="ollama",
    )
    parser.add_argument("--model")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face downloads for a missing cross-encoder model.",
    )
    return parser.parse_args(args)


@dataclass(frozen=True)
class ContextMemoryCaseResult:
    case_id: str
    passed: bool
    schema_valid: bool
    expected_selected_ids: list[str]
    returned_ids: list[str]
    relevant_returned_count: int
    irrelevant_returned_count: int
    inactive_leak_count: int
    false_none: bool
    latency_ms: float
    error: str | None = None


def _build_candidates(case: dict[str, Any]) -> tuple[RetrievedMemory, ...]:
    return tuple(
        RetrievedMemory(
            memory=Memory(
                id=item["id"],
                content=item["content"],
                memory_type=MemoryType(item["memory_type"]),
                confidence=float(item.get("confidence", 1.0)),
                status=MemoryStatus(item.get("status", "active")),
            ),
            score=float(item["candidate_score"]),
        )
        for item in case["candidates"]
    )


async def _score_case(
    case: dict[str, Any],
    reranker: MemoryReranker,
) -> ContextMemoryCaseResult:
    candidates = _build_candidates(case)
    started = perf_counter()
    error: str | None = None
    schema_valid = True
    try:
        outcome = await reranker.rerank(
            message=case["message"],
            intent=Intent(case["intent"]),
            candidates=candidates,
            limit=int(case["limit"]),
        )
        selected = outcome.memories
        schema_valid = outcome.succeeded
        error = outcome.error
    except Exception as exc:
        selected = ()
        schema_valid = False
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((perf_counter() - started) * 1000, 3)

    returned_ids = [item.memory.id for item in selected]
    expected = set(case["expected_selected_ids"])
    relevant_count = len(expected.intersection(returned_ids))
    irrelevant_count = sum(
        memory_id not in expected for memory_id in returned_ids
    )
    inactive_count = sum(
        item.memory.status is not MemoryStatus.ACTIVE for item in selected
    )
    false_none = bool(expected) and not returned_ids
    passed = (
        schema_valid
        and set(returned_ids) == expected
        and len(returned_ids) == len(expected)
        and inactive_count == 0
    )
    return ContextMemoryCaseResult(
        case_id=case["id"],
        passed=passed,
        schema_valid=schema_valid,
        expected_selected_ids=list(case["expected_selected_ids"]),
        returned_ids=returned_ids,
        relevant_returned_count=relevant_count,
        irrelevant_returned_count=irrelevant_count,
        inactive_leak_count=inactive_count,
        false_none=false_none,
        latency_ms=latency_ms,
        error=error,
    )


def _rate(numerator: int | float, denominator: int, empty: float) -> float:
    return empty if denominator == 0 else round(numerator / denominator, 4)


def _build_report(
    dataset: dict[str, Any],
    results: list[ContextMemoryCaseResult],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    expected_total = sum(len(item.expected_selected_ids) for item in results)
    returned_total = sum(len(item.returned_ids) for item in results)
    relevant_total = sum(item.relevant_returned_count for item in results)
    negative_results = [
        item for item in results if not item.expected_selected_ids
    ]
    positive_results = [item for item in results if item.expected_selected_ids]
    summary = {
        "case_pass_rate": _rate(
            sum(item.passed for item in results), len(results), 1.0
        ),
        "context_precision": _rate(relevant_total, returned_total, 1.0),
        "context_recall": _rate(relevant_total, expected_total, 1.0),
        "context_f1": 0.0,
        "valid_none_accuracy": _rate(
            sum(not item.returned_ids and item.schema_valid for item in negative_results),
            len(negative_results),
            1.0,
        ),
        "false_none_rate": _rate(
            sum(item.false_none for item in positive_results),
            len(positive_results),
            0.0,
        ),
        "irrelevant_inclusion_rate": _rate(
            sum(item.irrelevant_returned_count for item in results),
            returned_total,
            0.0,
        ),
        "inactive_memory_leakage_rate": _rate(
            sum(item.inactive_leak_count for item in results),
            returned_total,
            0.0,
        ),
        "schema_valid_response_rate": _rate(
            sum(item.schema_valid for item in results), len(results), 1.0
        ),
        "average_selection_size": _rate(returned_total, len(results), 0.0),
        "average_latency_ms": _rate(
            sum(item.latency_ms for item in results), len(results), 0.0
        ),
    }
    precision = summary["context_precision"]
    recall = summary["context_recall"]
    summary["context_f1"] = (
        0.0
        if precision + recall == 0
        else round(2 * precision * recall / (precision + recall), 4)
    )
    checks: dict[str, bool] = {}
    for metric, boundary in dataset["evaluation_contract"][
        "acceptance_gates"
    ].items():
        if "minimum" in boundary:
            checks[metric] = summary[metric] >= boundary["minimum"]
        elif "maximum" in boundary:
            checks[metric] = summary[metric] <= boundary["maximum"]
        else:
            raise ValueError(f"Gate for {metric} has no boundary.")
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
    reranker: MemoryReranker,
    model: str | None = None,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = [
        await _score_case(case, reranker) for case in dataset["cases"]
    ]
    report = _build_report(dataset, results, model=model)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(args: list[str] | None = None) -> int:
    from memory_engine.config import get_settings
    from memory_engine.llm import OllamaLLMClient
    from memory_engine.memory.reranker import (
        CrossEncoderMemoryReranker,
        LLMMemoryReranker,
        SentenceTransformerCrossEncoderScorer,
    )
    from memory_engine.prompting import PromptRepository
    from evaluation.clients import build_openai_evaluation_client
    from evaluation.usage import attach_usage_and_rewrite

    parsed = _parse_args(args)
    settings = get_settings()
    model = parsed.model or (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
        if parsed.provider == "cross_encoder"
        else (
            settings.context_reranking_model
            if parsed.provider == settings.context_reranking_provider
            else settings.semantic_model
        )
    )
    if parsed.provider == "cross_encoder":
        client = None
        reranker = CrossEncoderMemoryReranker(
            scorer=SentenceTransformerCrossEncoderScorer(
                model,
                local_files_only=not parsed.allow_model_download,
            ),
            relevance_threshold=parsed.threshold,
        )
    else:
        client = (
            OllamaLLMClient(
                model=model,
                base_url=settings.ollama_base_url,
                temperature=0.0,
                json_mode=True,
                timeout_seconds=settings.semantic_timeout_seconds,
            )
            if parsed.provider == "ollama"
            else build_openai_evaluation_client(model)
        )
        reranker = LLMMemoryReranker(
            llm_client=client,
            prompt_repository=PromptRepository(PROJECT_ROOT / "app/prompts"),
        )
    report = asyncio.run(
        run_evaluation(
            parsed.dataset,
            parsed.report,
            reranker=reranker,
            model=model,
        )
    )
    if client is not None:
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
    print(f"Report: {parsed.report}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
