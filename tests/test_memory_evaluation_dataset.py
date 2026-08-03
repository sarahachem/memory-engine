import json
from pathlib import Path

from memory_engine.models import MemoryType


PLANNER_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory.json"
)
PLANNER_HOLDOUT_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_holdout.json"
)
RETRIEVAL_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_retrieval.json"
)
RETRIEVAL_HARD_NEGATIVE_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_retrieval_hard_negatives.json"
)


def _assert_planner_dataset_contract(
    dataset: dict,
    *,
    minimum_cases: int,
    maximum_cases: int,
) -> None:
    cases = dataset["cases"]

    assert dataset["schema_version"] == "1.0"
    assert dataset["evaluation_target"] == "LLMMemoryPlanner"
    assert "content_faithfulness_rate" in (
        dataset["evaluation_contract"]["required_metrics"]
    )
    assert "generated_content" not in (
        dataset["evaluation_contract"]["ignored_for_exact_comparison"]
    )
    assert minimum_cases <= len(cases) <= maximum_cases
    assert len({case["id"] for case in cases}) == len(cases)

    valid_memory_types = {memory_type.value for memory_type in MemoryType}
    valid_operations = {"create", "update", "delete", "noop"}

    for case in cases:
        assert case["id"]
        assert case["description"]
        assert case["user_message"]
        assert case["tags"]

        for memory in case["active_memories"]:
            assert memory["content"]
            assert memory["memory_type"] in valid_memory_types
            assert 0.0 <= memory["confidence"] <= 1.0

        expected_operations = case["expected"]["operations"]
        assert expected_operations

        for operation in expected_operations:
            operation_name = operation["operation"]
            assert operation_name in valid_operations

            if operation_name == "create":
                assert operation["memory_type"] in valid_memory_types
                assert "memory_index" not in operation
            elif operation_name in {"update", "delete"}:
                assert 0 <= operation["memory_index"] < len(
                    case["active_memories"]
                )
                assert "memory_type" not in operation
            else:
                assert len(expected_operations) == 1
                assert set(operation) == {"operation"}

            if operation_name in {"create", "update"}:
                assertions = operation["content_assertions"]
                assert assertions["must_include"]
                assert isinstance(assertions["must_not_include"], list)
                assert set(assertions) == {
                    "must_include",
                    "must_not_include",
                }
            else:
                assert "content_assertions" not in operation


def test_memory_evaluation_dataset_contract() -> None:
    dataset = json.loads(
        PLANNER_DATASET_PATH.read_text(encoding="utf-8")
    )

    _assert_planner_dataset_contract(
        dataset,
        minimum_cases=20,
        maximum_cases=30,
    )


def test_memory_holdout_evaluation_dataset_contract() -> None:
    dataset = json.loads(
        PLANNER_HOLDOUT_DATASET_PATH.read_text(encoding="utf-8")
    )

    _assert_planner_dataset_contract(
        dataset,
        minimum_cases=10,
        maximum_cases=15,
    )
    assert {
        "false_delete_rate",
        "false_update_rate",
        "unsafe_mutation_rate",
        "duplicate_create_rate",
    } <= set(dataset["evaluation_contract"]["required_metrics"])
    assert any(
        "duplicate" in case["tags"]
        for case in dataset["cases"]
    )


def test_memory_holdout_case_ids_do_not_overlap_development_set() -> None:
    development_dataset = json.loads(
        PLANNER_DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout_dataset = json.loads(
        PLANNER_HOLDOUT_DATASET_PATH.read_text(encoding="utf-8")
    )

    development_ids = {
        case["id"]
        for case in development_dataset["cases"]
    }
    holdout_ids = {
        case["id"]
        for case in holdout_dataset["cases"]
    }

    assert development_ids.isdisjoint(holdout_ids)


def _assert_retrieval_dataset_contract(
    dataset: dict,
    *,
    minimum_cases: int,
) -> list[dict]:
    cases = dataset["cases"]

    assert dataset["schema_version"] == "1.0"
    assert dataset["evaluation_target"] == "SemanticCandidateRetriever"
    assert dataset["evaluation_contract"]["required_metrics"] == [
        "recall_at_k",
        "precision_at_k",
        "mean_reciprocal_rank",
        "critical_memory_miss_rate",
        "irrelevant_memory_inclusion_rate",
        "inactive_memory_leakage_rate",
    ]
    assert len(cases) >= minimum_cases
    assert len({case["id"] for case in cases}) == len(cases)

    valid_memory_types = {memory_type.value for memory_type in MemoryType}

    for case in cases:
        assert case["id"]
        assert case["description"]
        assert case["query"]
        assert case["tags"]
        assert case["limit"] >= 0

        memories = case["candidate_memories"]
        memory_ids = [memory["id"] for memory in memories]
        assert len(set(memory_ids)) == len(memory_ids)

        for memory in memories:
            assert memory["content"]
            assert memory["memory_type"] in valid_memory_types
            assert 0.0 <= memory["confidence"] <= 1.0
            assert memory.get("status", "active") in {
                "active",
                "invalidated",
                "deleted",
            }

        relevant_ids = set(case["expected_relevant_ids"])
        critical_ids = set(case["critical_relevant_ids"])
        active_ids = {
            memory["id"]
            for memory in memories
            if memory.get("status", "active") == "active"
        }

        assert relevant_ids <= active_ids
        assert critical_ids <= relevant_ids

    return cases


def test_memory_retrieval_evaluation_dataset_contract() -> None:
    dataset = json.loads(
        RETRIEVAL_DATASET_PATH.read_text(encoding="utf-8")
    )
    _assert_retrieval_dataset_contract(dataset, minimum_cases=10)


def test_memory_retrieval_hard_negative_dataset_contract() -> None:
    """
    Structural contract plus a corpus-size floor: this dataset exists
    specifically to test retrieval at realistic scale, so nothing should
    let it silently regress back to the 1-3 candidate toy cases that
    motivated it. See docs/architecture/runtime-memory-decisions.md §7 —
    "tiny synthetic corpora" prove small-corpus ranking, not production
    recall.
    """
    dataset = json.loads(
        RETRIEVAL_HARD_NEGATIVE_DATASET_PATH.read_text(encoding="utf-8")
    )
    cases = _assert_retrieval_dataset_contract(dataset, minimum_cases=10)

    corpus_sizes = [len(case["candidate_memories"]) for case in cases]
    assert min(corpus_sizes) >= 10
    assert sum(corpus_sizes) / len(corpus_sizes) >= 15

    # At least one no-relevant-memory case is required so precision under
    # a large, topically dense corpus is actually exercised, not just
    # recall.
    assert any(not case["expected_relevant_ids"] for case in cases)
    # At least one case must carry an invalidated or deleted decoy so a
    # lexically-close but inactive memory can't silently leak through.
    assert any(
        any(
            memory.get("status", "active") != "active"
            for memory in case["candidate_memories"]
        )
        for case in cases
    )
