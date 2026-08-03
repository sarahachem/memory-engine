import json
from pathlib import Path

from memory_engine.models import MemoryType


DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "datasets"
    / "memory_pipeline.json"
)


def test_memory_pipeline_dataset_scoring_contract() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    contract = dataset["evaluation_contract"]

    assert "memory_type_accuracy" in contract["required_metrics"]
    assert not contract["scoring_notes"][
        "type_only_mismatch_is_unsafe"
    ]

    valid_types = {item.value for item in MemoryType}
    ambiguous_cases = 0

    for case in dataset["cases"]:
        for expected in case["expected"]["active_memories"]:
            accepted_types = expected.get(
                "accepted_memory_types",
                [expected["memory_type"]],
            )
            assert accepted_types
            assert set(accepted_types) <= valid_types

            assertions = expected["content_assertions"]
            assert (
                assertions.get("must_include")
                or assertions.get("must_include_any")
            )

            for alternative in assertions.get(
                "must_include_any",
                [],
            ):
                assert alternative

            if "accepted_memory_types" in expected:
                ambiguous_cases += 1

    assert ambiguous_cases >= 1
