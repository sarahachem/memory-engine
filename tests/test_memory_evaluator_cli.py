from pathlib import Path

from evaluation.evaluators.memory_evaluator import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORT_PATH,
    _parse_args,
)


def test_memory_evaluator_cli_uses_default_paths() -> None:
    args = _parse_args([])

    assert args.dataset == DEFAULT_DATASET_PATH
    assert args.report == DEFAULT_REPORT_PATH


def test_memory_evaluator_cli_accepts_explicit_paths() -> None:
    args = _parse_args(
        [
            "--dataset",
            "evaluation/datasets/memory_holdout.json",
            "--report",
            "evaluation/reports/memory_holdout.json",
        ]
    )

    assert args.dataset == Path(
        "evaluation/datasets/memory_holdout.json"
    )
    assert args.report == Path(
        "evaluation/reports/memory_holdout.json"
    )
