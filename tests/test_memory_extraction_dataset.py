import json
from pathlib import Path

from memory_engine.models import MemoryType


PROJECT_ROOT = Path(__file__).parents[1]
DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction.json"
)
HOLDOUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction_holdout.json"
)
HOLDOUT_V2_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction_holdout_v2.json"
)
HOLDOUT_V3_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction_holdout_v3.json"
)
HOLDOUT_V4_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction_holdout_v4.json"
)
HOLDOUT_V5_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction_holdout_v5.json"
)


def _assert_contract(
    dataset: dict,
    *,
    minimum_cases: int,
    maximum_cases: int,
) -> None:
    assert dataset["schema_version"] == "1.0"
    assert dataset["evaluation_target"] == "LLMMemoryFactExtractor"
    assert minimum_cases <= len(dataset["cases"]) <= maximum_cases
    assert len({case["id"] for case in dataset["cases"]}) == len(
        dataset["cases"]
    )

    required_metrics = set(
        dataset["evaluation_contract"]["required_metrics"]
    )
    assert {
        "fact_precision",
        "fact_recall",
        "compound_fact_recall",
        "evidence_grounding_rate",
        "false_memory_rate",
    } <= required_metrics
    assert set(
        dataset["evaluation_contract"]["acceptance_gates"]
    ) <= required_metrics

    valid_types = {memory_type.value for memory_type in MemoryType}
    for case in dataset["cases"]:
        assert case["id"]
        assert case["description"]
        assert case["user_message"]
        assert case["tags"]
        for fact in case["expected"]["facts"]:
            assert fact["kind"] in {"assertion", "invalidation"}
            assert fact["memory_type"] in valid_types
            assert fact["accepted_evidence"]
            for evidence in fact["accepted_evidence"]:
                assert (
                    evidence.casefold()
                    in case["user_message"].casefold()
                )
            assertions = fact["content_assertions"]
            assert assertions["must_include"]
            assert isinstance(assertions["must_not_include"], list)
            assert set(assertions) <= {
                "must_include",
                "must_include_any",
                "must_not_include",
            }


def test_memory_extraction_development_dataset_contract() -> None:
    dataset = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    _assert_contract(dataset, minimum_cases=20, maximum_cases=30)
    assert any("compound" in case["tags"] for case in dataset["cases"])
    assert any(
        not case["expected"]["facts"] for case in dataset["cases"]
    )


def test_memory_extraction_holdout_dataset_contract() -> None:
    dataset = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    _assert_contract(dataset, minimum_cases=10, maximum_cases=15)
    assert "holdout" in dataset["description"].casefold()


def test_memory_extraction_holdout_v2_dataset_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V2_PATH.read_text(encoding="utf-8")
    )
    _assert_contract(dataset, minimum_cases=10, maximum_cases=15)
    assert "second unseen holdout" in dataset[
        "description"
    ].casefold()


def test_memory_extraction_holdout_v3_dataset_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V3_PATH.read_text(encoding="utf-8")
    )
    _assert_contract(dataset, minimum_cases=10, maximum_cases=15)
    assert "third unseen" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()


def test_memory_extraction_holdout_v4_dataset_contract() -> None:
    dataset = json.loads(
        HOLDOUT_V4_PATH.read_text(encoding="utf-8")
    )
    _assert_contract(dataset, minimum_cases=10, maximum_cases=15)
    assert "fourth unseen" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()
    assert "semantic_judge_valid_rate" in dataset[
        "evaluation_contract"
    ]["required_metrics"]


def test_memory_extraction_holdout_v5_dataset_contract() -> None:
    dataset = json.loads(HOLDOUT_V5_PATH.read_text(encoding="utf-8"))
    _assert_contract(dataset, minimum_cases=10, maximum_cases=15)
    assert "fifth unseen" in dataset["description"].casefold()
    assert "never tune" in dataset["description"].casefold()
    assert "gpt-5.4 mini" in dataset["description"].casefold()
    assert "semantic_judge_valid_rate" in dataset[
        "evaluation_contract"
    ]["required_metrics"]


def test_extraction_dataset_ids_do_not_overlap() -> None:
    development = json.loads(
        DEVELOPMENT_PATH.read_text(encoding="utf-8")
    )
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    holdout_v2 = json.loads(
        HOLDOUT_V2_PATH.read_text(encoding="utf-8")
    )
    holdout_v3 = json.loads(
        HOLDOUT_V3_PATH.read_text(encoding="utf-8")
    )
    holdout_v4 = json.loads(
        HOLDOUT_V4_PATH.read_text(encoding="utf-8")
    )
    holdout_v5 = json.loads(
        HOLDOUT_V5_PATH.read_text(encoding="utf-8")
    )

    dataset_ids = [
        {case["id"] for case in dataset["cases"]}
        for dataset in (
            development,
            holdout,
            holdout_v2,
            holdout_v3,
            holdout_v4,
            holdout_v5,
        )
    ]
    for index, ids in enumerate(dataset_ids):
        for other_ids in dataset_ids[index + 1:]:
            assert ids.isdisjoint(other_ids)


def test_recurrence_boundary_has_development_coverage() -> None:
    dataset = json.loads(
        DEVELOPMENT_PATH.read_text(encoding="utf-8")
    )
    recurrence_cases = [
        case
        for case in dataset["cases"]
        if "recurrence_contrast" in case["tags"]
    ]

    assert any(
        not case["expected"]["facts"] for case in recurrence_cases
    )
    assert any(
        case["expected"]["facts"] for case in recurrence_cases
    )
