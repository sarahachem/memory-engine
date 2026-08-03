import asyncio
import json

from memory_engine.memory.reconciler import (
    CreateDecision,
    MemoryReconciler,
    UpdateDecision,
)
from evaluation.evaluators.memory_reconciliation_evaluator import (
    DEFAULT_DATASET_PATH,
    DEFAULT_HOLDOUT_PATH,
    DEFAULT_HOLDOUT_V2_PATH,
    DEFAULT_HOLDOUT_V3_PATH,
    DEFAULT_HOLDOUT_V4_PATH,
    _build_requests,
    _content_passes,
    run_evaluation,
)


def test_forbidden_content_requires_contiguous_reversed_proposition() -> None:
    assertions = {
        "must_include": ["phone calls"],
        "must_not_include": ["prefers text messages"],
    }

    assert _content_passes(
        "The user prefers phone calls to text messages.", assertions
    )
    assert not _content_passes(
        "The user prefers text messages to phone calls.", assertions
    )


class BatchReconciler(MemoryReconciler):
    def __init__(self, decisions) -> None:
        self.decisions = tuple(decisions)
        self.calls = []

    async def reconcile(self, fact, candidate_memories):
        raise AssertionError("Evaluator must use reconcile_many")

    async def reconcile_many(self, requests):
        self.calls.append(requests)
        return self.decisions


def _dataset() -> dict:
    return {
        "schema_version": "1.0",
        "evaluation_target": "LLMMemoryReconciler.reconcile_many",
        "description": "Fixture development dataset.",
        "evaluation_contract": {
            "acceptance_gates": {
                "schema_valid_response_rate": {"minimum": 1.0},
                "operation_accuracy": {"minimum": 1.0},
                "target_index_accuracy": {"minimum": 1.0},
                "false_destructive_mutation_rate": {"maximum": 0.0},
            }
        },
        "cases": [
            {
                "id": "fixture",
                "facts": [
                    {
                        "fact": {
                            "kind": "assertion",
                            "content": "The user now lives in Berlin.",
                            "memory_type": "personal_fact",
                            "evidence": "I now live in Berlin",
                            "confidence": 0.99,
                        },
                        "candidate_memories": [
                            {
                                "id": "location",
                                "content": "The user lives in Munich.",
                                "memory_type": "personal_fact",
                                "confidence": 0.95,
                            }
                        ],
                        "expected": {
                            "action": "update",
                            "target_memory_ids": ["location"],
                            "content_assertions": {
                                "must_include": ["Berlin"],
                                "must_not_include": ["Munich"],
                            },
                        },
                    },
                    {
                        "fact": {
                            "kind": "assertion",
                            "content": "The user wants to learn Japanese.",
                            "memory_type": "goal",
                            "evidence": "I want to learn Japanese",
                            "confidence": 0.98,
                        },
                        "candidate_memories": [],
                        "expected": {"action": "create"},
                    },
                ],
            }
        ],
    }


def test_evaluator_scores_batch_actions_targets_content_and_isolation(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    dataset_path.write_text(json.dumps(_dataset()), encoding="utf-8")
    reconciler = BatchReconciler(
        (
            UpdateDecision(
                action="update",
                memory_index=0,
                content="The user now lives in Berlin.",
                confidence=0.97,
                explanation="Location replacement.",
            ),
            CreateDecision(
                action="create",
                confidence=0.96,
                explanation="New goal.",
            ),
        )
    )

    report = asyncio.run(
        run_evaluation(
            dataset_path,
            report_path,
            reconciler=reconciler,
            model="fixture-model",
        )
    )

    assert len(reconciler.calls) == 1
    assert len(reconciler.calls[0]) == 2
    assert report["acceptance_gates"]["passed"]
    assert report["summary"]["operation_accuracy"] == 1.0
    assert report["summary"]["target_index_accuracy"] == 1.0
    assert report["summary"]["update_content_faithfulness_rate"] == 1.0
    assert report["summary"]["fact_isolation_rate"] == 1.0
    assert report["summary"]["false_create_rate"] == 0.0
    assert report["cases"][0]["actual_decisions"][0][
        "target_memory_id"
    ] == "location"


def test_wrong_local_target_is_an_unsafe_isolation_failure(tmp_path) -> None:
    dataset = _dataset()
    dataset["cases"][0]["facts"][0]["candidate_memories"].append(
        {
            "id": "distractor",
            "content": "The user works as a designer.",
            "memory_type": "personal_fact",
            "confidence": 0.91,
        }
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = asyncio.run(
        run_evaluation(
            dataset_path,
            tmp_path / "report.json",
            reconciler=BatchReconciler(
                (
                    UpdateDecision(
                        action="update",
                        memory_index=1,
                        content="The user now lives in Berlin.",
                        confidence=0.97,
                        explanation="Wrong target.",
                    ),
                    CreateDecision(
                        action="create",
                        confidence=0.96,
                        explanation="New goal.",
                    ),
                )
            ),
        )
    )

    assert not report["cases"][0]["passed"]
    assert report["summary"]["target_index_accuracy"] == 0.0
    assert report["summary"]["false_destructive_mutation_rate"] == 0.5
    assert report["summary"]["fact_isolation_rate"] == 0.5


def test_development_dataset_covers_batch_risks() -> None:
    dataset = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    actions = {
        expected_action
        for case in cases
        for fact in case["facts"]
        for expected_action in fact["expected"].get(
            "accepted_actions",
            [fact["expected"]["action"]],
        )
    }

    assert "Development" in dataset["description"]
    assert len(cases) >= 14
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(len(case["facts"]) > 1 for case in cases) >= 10
    assert actions == {"create", "update", "invalidate", "noop"}
    assert any("batch_constraints" in case for case in cases)
    assert any(
        len(fact["candidate_memories"]) > 1
        for case in cases
        for fact in case["facts"]
    )
    assert all(_build_requests(case) for case in cases)


def test_reconciliation_holdout_contract_and_isolation() -> None:
    development = json.loads(
        DEFAULT_DATASET_PATH.read_text(encoding="utf-8")
    )
    holdout = json.loads(
        DEFAULT_HOLDOUT_PATH.read_text(encoding="utf-8")
    )
    cases = holdout["cases"]
    actions = {
        action
        for case in cases
        for fact in case["facts"]
        for action in fact["expected"].get(
            "accepted_actions",
            [fact["expected"]["action"]],
        )
    }

    assert holdout["schema_version"] == "1.0"
    assert holdout["evaluation_target"] == (
        "LLMMemoryReconciler.reconcile_many"
    )
    assert "untouched" in holdout["description"].casefold()
    assert "never tune" in holdout["description"].casefold()
    assert 10 <= len(cases) <= 15
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(len(case["facts"]) > 1 for case in cases) >= 7
    assert actions == {"create", "update", "invalidate", "noop"}
    assert any("batch_constraints" in case for case in cases)
    assert any(
        len(fact["candidate_memories"]) > 1
        for case in cases
        for fact in case["facts"]
    )
    assert all(_build_requests(case) for case in cases)

    development_ids = {case["id"] for case in development["cases"]}
    holdout_ids = {case["id"] for case in cases}
    assert development_ids.isdisjoint(holdout_ids)


def test_reconciliation_holdout_v2_contract_and_isolation() -> None:
    datasets = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            DEFAULT_DATASET_PATH,
            DEFAULT_HOLDOUT_PATH,
            DEFAULT_HOLDOUT_V2_PATH,
        )
    ]
    holdout = datasets[-1]
    cases = holdout["cases"]
    actions = {
        action
        for case in cases
        for fact in case["facts"]
        for action in fact["expected"].get(
            "accepted_actions",
            [fact["expected"]["action"]],
        )
    }

    assert "second untouched" in holdout["description"].casefold()
    assert "never tune" in holdout["description"].casefold()
    assert 10 <= len(cases) <= 15
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(len(case["facts"]) > 1 for case in cases) >= 5
    assert actions == {"create", "update", "invalidate", "noop"}
    assert any("batch_constraints" in case for case in cases)
    assert all(_build_requests(case) for case in cases)

    id_sets = [
        {case["id"] for case in dataset["cases"]}
        for dataset in datasets
    ]
    for index, ids in enumerate(id_sets):
        for other_ids in id_sets[index + 1 :]:
            assert ids.isdisjoint(other_ids)


def test_reconciliation_holdout_v3_contract_and_isolation() -> None:
    datasets = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            DEFAULT_DATASET_PATH,
            DEFAULT_HOLDOUT_PATH,
            DEFAULT_HOLDOUT_V2_PATH,
            DEFAULT_HOLDOUT_V3_PATH,
        )
    ]
    holdout = datasets[-1]
    cases = holdout["cases"]
    actions = {
        action
        for case in cases
        for fact in case["facts"]
        for action in fact["expected"].get(
            "accepted_actions", [fact["expected"]["action"]]
        )
    }

    assert "third untouched" in holdout["description"].casefold()
    assert "never tune" in holdout["description"].casefold()
    assert "gpt-5.4 mini" in holdout["description"].casefold()
    assert 10 <= len(cases) <= 15
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(len(case["facts"]) > 1 for case in cases) >= 3
    assert actions == {"create", "update", "invalidate", "noop"}
    assert any("batch_constraints" in case for case in cases)
    assert all(_build_requests(case) for case in cases)

    id_sets = [
        {case["id"] for case in dataset["cases"]}
        for dataset in datasets
    ]
    for index, ids in enumerate(id_sets):
        for other_ids in id_sets[index + 1 :]:
            assert ids.isdisjoint(other_ids)


def test_reconciliation_holdout_v4_contract_and_isolation() -> None:
    paths = (
        DEFAULT_DATASET_PATH,
        DEFAULT_HOLDOUT_PATH,
        DEFAULT_HOLDOUT_V2_PATH,
        DEFAULT_HOLDOUT_V3_PATH,
        DEFAULT_HOLDOUT_V4_PATH,
    )
    datasets = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    holdout = datasets[-1]
    cases = holdout["cases"]
    actions = {
        action
        for case in cases
        for fact in case["facts"]
        for action in fact["expected"].get(
            "accepted_actions", [fact["expected"]["action"]]
        )
    }
    assert "fourth untouched" in holdout["description"].casefold()
    assert "never tune" in holdout["description"].casefold()
    assert "gpt-5.6 terra" in holdout["description"].casefold()
    assert 10 <= len(cases) <= 15
    assert actions == {"create", "update", "invalidate", "noop"}
    assert any("batch_constraints" in case for case in cases)
    assert all(_build_requests(case) for case in cases)
    id_sets = [{case["id"] for case in data["cases"]} for data in datasets]
    for index, ids in enumerate(id_sets):
        for other_ids in id_sets[index + 1 :]:
            assert ids.isdisjoint(other_ids)
