import json
from pathlib import Path

import pytest

from memory_engine.memory.reranker import FakeMemoryReranker
from evaluation.evaluators.context_memory_evaluator import (
    DEFAULT_DATASET_PATH,
    _build_report,
    _parse_args,
    _score_case,
)


def _case(expected: list[str] | None = None) -> dict:
    return {
        "id": "case",
        "description": "Fixture",
        "tags": ["fixture"],
        "message": "Should I continue learning Japanese?",
        "intent": "decision_support",
        "limit": 2,
        "candidates": [
            {
                "id": "japanese",
                "content": "The user wants to learn Japanese.",
                "memory_type": "goal",
                "candidate_score": 0.8,
            },
            {
                "id": "spanish",
                "content": "The user wants to learn Spanish.",
                "memory_type": "goal",
                "candidate_score": 0.7,
            },
        ],
        "expected_selected_ids": (
            ["japanese"] if expected is None else expected
        ),
    }


@pytest.mark.asyncio
async def test_context_evaluator_scores_precise_selection() -> None:
    case = _case()
    result = await _score_case(
        case,
        FakeMemoryReranker({case["message"]: ["japanese"]}),
    )

    assert result.passed
    assert result.schema_valid
    assert result.relevant_returned_count == 1
    assert result.irrelevant_returned_count == 0


@pytest.mark.asyncio
async def test_context_evaluator_distinguishes_valid_none_from_failure() -> None:
    case = _case(expected=[])
    valid_none = await _score_case(
        case,
        FakeMemoryReranker({case["message"]: []}),
    )
    failed = await _score_case(
        case,
        FakeMemoryReranker({}),
    )

    assert valid_none.passed
    assert valid_none.schema_valid
    assert valid_none.returned_ids == []
    assert not failed.passed
    assert not failed.schema_valid
    assert failed.returned_ids == []


@pytest.mark.asyncio
async def test_context_report_computes_precision_focused_metrics() -> None:
    positive = _case()
    positive_result = await _score_case(
        positive,
        FakeMemoryReranker({positive["message"]: ["japanese"]}),
    )
    negative = {
        **_case(expected=[]),
        "id": "negative",
        "message": "Hello",
    }
    negative_result = await _score_case(
        negative,
        FakeMemoryReranker({"Hello": []}),
    )
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "MemoryReranker",
        "description": "Fixture",
        "evaluation_contract": {
            "acceptance_gates": {
                "context_precision": {"minimum": 0.95},
                "valid_none_accuracy": {"minimum": 0.95},
            }
        },
    }

    report = _build_report(
        dataset,
        [positive_result, negative_result],
        model="fake",
    )

    assert report["summary"]["context_precision"] == 1.0
    assert report["summary"]["context_recall"] == 1.0
    assert report["summary"]["context_f1"] == 1.0
    assert report["summary"]["valid_none_accuracy"] == 1.0
    assert report["summary"]["false_none_rate"] == 0.0
    assert report["acceptance_gates"]["passed"]


def test_context_memory_dataset_contract() -> None:
    dataset = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    assert dataset["evaluation_target"] == "MemoryReranker"
    assert 10 <= len(cases) <= 30
    assert len({case["id"] for case in cases}) == len(cases)
    assert any(not case["expected_selected_ids"] for case in cases)
    assert any(len(case["expected_selected_ids"]) > 1 for case in cases)
    for case in cases:
        candidate_ids = {item["id"] for item in case["candidates"]}
        assert set(case["expected_selected_ids"]) <= candidate_ids
        assert len(case["expected_selected_ids"]) <= case["limit"]


def test_context_memory_holdouts_use_supported_record_contracts() -> None:
    dataset_dir = DEFAULT_DATASET_PATH.parent
    for path in dataset_dir.glob("context_memory_holdout*.json"):
        dataset = json.loads(path.read_text(encoding="utf-8"))
        assert dataset["evaluation_target"] == "MemoryReranker"
        for case in dataset["cases"]:
            for candidate in case["candidates"]:
                assert candidate.get("status", "active") in {
                    "active",
                    "invalidated",
                }, f"Unsupported status in {Path(path).name}:{case['id']}"


def test_context_evaluator_accepts_explicit_provider_and_model() -> None:
    parsed = _parse_args(
        ["--provider", "ollama", "--model", "qwen3:4b"]
    )

    assert parsed.provider == "ollama"
    assert parsed.model == "qwen3:4b"


def test_context_evaluator_accepts_cross_encoder_threshold() -> None:
    parsed = _parse_args(
        [
            "--provider",
            "cross_encoder",
            "--model",
            "cross-encoder/test",
            "--threshold",
            "-2.5",
            "--allow-model-download",
        ]
    )

    assert parsed.provider == "cross_encoder"
    assert parsed.model == "cross-encoder/test"
    assert parsed.threshold == -2.5
    assert parsed.allow_model_download
