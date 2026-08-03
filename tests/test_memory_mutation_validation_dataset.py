import json
from pathlib import Path

from memory_engine.models import MemoryType


DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation.json"
)
HOLDOUT_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation_holdout.json"
)
HOLDOUT_V2_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation_holdout_v2.json"
)
HOLDOUT_V3_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation_holdout_v3.json"
)
HOLDOUT_V4_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation_holdout_v4.json"
)
HOLDOUT_V5_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_mutation_validation_holdout_v5.json"
)


def _assert_dataset_contract(dataset: dict) -> None:
    cases = dataset["cases"]

    assert dataset["evaluation_target"] == (
        "LLMMemoryMutationValidator"
    )
    assert dataset["evaluation_contract"]["required_metrics"] == [
        "schema_valid_response_rate",
            "decision_accuracy",
            "unsafe_approval_rate",
            "safe_approval_rate",
            "safe_rejection_rate",
    ]
    assert len(cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["operation"] for case in cases} == {
        "update",
        "delete",
    }
    assert {case["expected"]["approved"] for case in cases} == {
        True,
        False,
    }

    valid_types = {item.value for item in MemoryType}
    for case in cases:
        assert case["target_memory"]["memory_type"] in valid_types
        assert bool(case.get("proposed_content")) == (
            case["operation"] == "update"
        )


def test_memory_mutation_validation_dataset_contract() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    _assert_dataset_contract(dataset)


def test_memory_mutation_validation_holdout_contract() -> None:
    dataset = json.loads(
        HOLDOUT_DATASET_PATH.read_text(encoding="utf-8")
    )

    _assert_dataset_contract(dataset)
    assert len(dataset["cases"]) == 12

    decisions = [
        case["expected"]["approved"]
        for case in dataset["cases"]
    ]
    assert decisions.count(True) == 6
    assert decisions.count(False) == 6


def test_memory_mutation_validation_holdout_v2_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V2_DATASET_PATH.read_text(encoding="utf-8")
    )

    _assert_dataset_contract(dataset)
    assert "second untouched" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()
    assert len(dataset["cases"]) == 12

    decisions = [
        case["expected"]["approved"]
        for case in dataset["cases"]
    ]
    assert decisions.count(True) == 6
    assert decisions.count(False) == 6


def test_memory_mutation_validation_holdout_v3_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V3_DATASET_PATH.read_text(encoding="utf-8")
    )

    _assert_dataset_contract(dataset)
    assert "third untouched" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()
    assert len(dataset["cases"]) == 12
    decisions = [case["expected"]["approved"] for case in dataset["cases"]]
    assert decisions.count(True) == 6
    assert decisions.count(False) == 6


def test_memory_mutation_validation_holdout_v4_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V4_DATASET_PATH.read_text(encoding="utf-8")
    )

    _assert_dataset_contract(dataset)
    assert "fourth untouched" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()
    assert "gpt-5.4 mini" in dataset["description"].casefold()
    assert len(dataset["cases"]) == 12
    decisions = [case["expected"]["approved"] for case in dataset["cases"]]
    assert decisions.count(True) == 6
    assert decisions.count(False) == 6


def test_memory_mutation_validation_holdout_v5_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V5_DATASET_PATH.read_text(encoding="utf-8")
    )
    _assert_dataset_contract(dataset)
    assert "fifth untouched" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()
    assert "gpt-5.6 terra" in dataset["description"].casefold()
    assert len(dataset["cases"]) == 12
    decisions = [case["expected"]["approved"] for case in dataset["cases"]]
    assert decisions.count(True) == 6
    assert decisions.count(False) == 6


def test_validator_holdout_ids_do_not_overlap_development_set() -> None:
    development = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout = json.loads(
        HOLDOUT_DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout_v2 = json.loads(
        HOLDOUT_V2_DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout_v3 = json.loads(
        HOLDOUT_V3_DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout_v4 = json.loads(
        HOLDOUT_V4_DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout_v5 = json.loads(
        HOLDOUT_V5_DATASET_PATH.read_text(encoding="utf-8")
    )

    development_ids = {
        case["id"]
        for case in development["cases"]
    }
    holdout_ids = {
        case["id"]
        for case in holdout["cases"]
    }
    holdout_v2_ids = {
        case["id"]
        for case in holdout_v2["cases"]
    }
    holdout_v3_ids = {
        case["id"]
        for case in holdout_v3["cases"]
    }
    holdout_v4_ids = {
        case["id"]
        for case in holdout_v4["cases"]
    }
    holdout_v5_ids = {
        case["id"]
        for case in holdout_v5["cases"]
    }

    assert development_ids.isdisjoint(holdout_ids)
    assert development_ids.isdisjoint(holdout_v2_ids)
    assert holdout_ids.isdisjoint(holdout_v2_ids)
    assert development_ids.isdisjoint(holdout_v3_ids)
    assert holdout_ids.isdisjoint(holdout_v3_ids)
    assert holdout_v2_ids.isdisjoint(holdout_v3_ids)
    assert development_ids.isdisjoint(holdout_v4_ids)
    assert holdout_ids.isdisjoint(holdout_v4_ids)
    assert holdout_v2_ids.isdisjoint(holdout_v4_ids)
    assert holdout_v3_ids.isdisjoint(holdout_v4_ids)
    for prior_ids in (
        development_ids,
        holdout_ids,
        holdout_v2_ids,
        holdout_v3_ids,
        holdout_v4_ids,
    ):
        assert prior_ids.isdisjoint(holdout_v5_ids)
