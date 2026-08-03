from pathlib import Path

import pytest

from memory_engine.memory.retriever import MemoryRetriever, RetrievedMemory
from evaluation.evaluators.memory_retrieval_evaluator import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORT_PATH,
    _build_report,
    _parse_args,
    _score_case,
)


class IdRetriever(MemoryRetriever):
    def __init__(self, returned_ids: list[str]) -> None:
        self.returned_ids = returned_ids

    async def retrieve(
        self,
        query,
        candidate_memories,
        limit=5,
    ):
        by_id = {
            memory.id: memory for memory in candidate_memories
        }
        return tuple(
            RetrievedMemory(memory=by_id[memory_id], score=0.9)
            for memory_id in self.returned_ids[:limit]
        )


def _case() -> dict:
    return {
        "id": "case-1",
        "description": "Fixture.",
        "tags": ["semantic"],
        "query": "How can I stay consistent with exercise?",
        "candidate_memories": [
            {
                "id": "pattern",
                "content": "The user skips workouts when work is busy.",
                "memory_type": "recurring_pattern",
                "confidence": 0.95,
            },
            {
                "id": "language",
                "content": "The user wants to learn German.",
                "memory_type": "goal",
                "confidence": 0.95,
            },
            {
                "id": "inactive",
                "content": "The user trains for a marathon.",
                "memory_type": "goal",
                "confidence": 0.95,
                "status": "invalidated",
            },
        ],
        "expected_relevant_ids": ["pattern"],
        "critical_relevant_ids": ["pattern"],
        "limit": 2,
    }


def test_retrieval_evaluator_cli_paths() -> None:
    defaults = _parse_args([])
    assert defaults.dataset == DEFAULT_DATASET_PATH
    assert defaults.report == DEFAULT_REPORT_PATH

    explicit = _parse_args(
        ["--dataset", "data.json", "--report", "report.json"]
    )
    assert explicit.dataset == Path("data.json")
    assert explicit.report == Path("report.json")


@pytest.mark.asyncio
async def test_score_case_accepts_only_relevant_active_memory() -> None:
    result = await _score_case(_case(), IdRetriever(["pattern"]))

    assert result.passed
    assert result.relevant_returned_count == 1
    assert result.reciprocal_rank == 1.0
    assert result.inactive_leak_count == 0


@pytest.mark.asyncio
async def test_score_case_detects_irrelevant_and_inactive_results() -> None:
    result = await _score_case(
        _case(),
        IdRetriever(["inactive", "language"]),
    )

    assert not result.passed
    assert result.critical_miss_count == 1
    assert result.irrelevant_returned_count == 2
    assert result.inactive_leak_count == 1


@pytest.mark.asyncio
async def test_report_computes_retrieval_metrics_and_gates() -> None:
    case = _case()
    result = await _score_case(case, IdRetriever(["pattern"]))
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "SemanticCandidateRetriever",
        "description": "Fixture.",
        "evaluation_contract": {
            "acceptance_gates": {
                "recall_at_k": {"minimum": 0.9},
                "inactive_memory_leakage_rate": {"maximum": 0.0},
            }
        },
        "cases": [case],
    }

    report = _build_report(
        dataset,
        [result],
        model="test-embedding",
        technical_floor=0.0,
    )

    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["precision_at_k"] == 1.0
    assert report["summary"]["mean_reciprocal_rank"] == 1.0
    assert report["summary"]["inactive_memory_leakage_rate"] == 0.0
    assert report["acceptance_gates"]["passed"]
