import asyncio
import json
from pathlib import Path

import pytest

from evaluation.evaluators.memory_extraction_evaluator import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORT_PATH,
    ExtractionCaseResult,
    LLMMemoryExtractionJudge,
    MemoryExtractionSemanticJudgment,
    SemanticFactGroup,
    _build_report,
    _evidence_matches,
    _parse_args,
    _score_case,
    _validate_semantic_judgment,
)


class RecordingLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []

    async def generate(self, messages, response_schema=None):
        self.calls.append((messages, response_schema))
        return json.dumps(self.response)


def test_evidence_scoring_ignores_only_terminal_sentence_punctuation() -> None:
    expected = {"accepted_evidence": ["I prefer written feedback"]}

    assert _evidence_matches("I prefer written feedback.", expected)
    assert not _evidence_matches("I prefer direct feedback.", expected)


def _case(
    *,
    expected_facts: list[dict] | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": "case-1",
        "description": "Metric fixture.",
        "tags": tags or ["compound"],
        "user_message": (
            "I gave up opening a bakery and I prefer direct feedback"
        ),
        "expected": {
            "facts": expected_facts
            if expected_facts is not None
            else [
                {
                    "kind": "invalidation",
                    "memory_type": "goal",
                    "accepted_evidence": [
                        "I gave up opening a bakery"
                    ],
                    "content_assertions": {
                        "must_include": ["no longer", "open", "bakery"],
                        "must_not_include": ["feedback"],
                    },
                },
                {
                    "kind": "assertion",
                    "memory_type": "preference",
                    "accepted_evidence": [
                        "I prefer direct feedback"
                    ],
                    "content_assertions": {
                        "must_include": ["prefer", "direct feedback"],
                        "must_not_include": ["bakery"],
                    },
                },
            ],
        },
    }


def _actual_facts() -> list[dict]:
    return [
        {
            "kind": "invalidation",
            "content": "The user no longer wants to open a bakery.",
            "memory_type": "goal",
            "evidence": "I gave up opening a bakery",
            "confidence": 0.99,
        },
        {
            "kind": "assertion",
            "content": "The user prefers direct feedback.",
            "memory_type": "preference",
            "evidence": "I prefer direct feedback",
            "confidence": 0.98,
        },
    ]


def test_extraction_evaluator_cli_paths() -> None:
    defaults = _parse_args([])
    assert defaults.dataset == DEFAULT_DATASET_PATH
    assert defaults.report == DEFAULT_REPORT_PATH

    explicit = _parse_args(
        [
            "--dataset",
            "holdout.json",
            "--report",
            "report.json",
        ]
    )
    assert explicit.dataset == Path("holdout.json")
    assert explicit.report == Path("report.json")


def test_score_case_accepts_complete_atomic_compound_extraction() -> None:
    result = _score_case(
        _case(),
        _actual_facts(),
        schema_valid=True,
    )

    assert result.passed
    assert result.strict_matches == 2
    assert result.content_matches == 2
    assert result.kind_matches == 2
    assert result.memory_type_matches == 2
    assert result.evidence_matches == 2
    assert result.grounded_evidence_count == 2


def test_score_case_accepts_faithful_invalidation_paraphrase() -> None:
    case = {
        "id": "invalidation-paraphrase",
        "tags": ["invalidation"],
        "user_message": (
            "Running a marathon is no longer one of my goals."
        ),
        "expected": {
            "facts": [
                {
                    "kind": "invalidation",
                    "memory_type": "goal",
                    "accepted_evidence": [
                        "Running a marathon is no longer one of my goals"
                    ],
                    "content_assertions": {
                        "must_include": ["marathon"],
                        "must_include_any": [
                            ["no longer", "goal"],
                            ["no longer", "want", "run"],
                        ],
                        "must_not_include": [],
                    },
                }
            ]
        },
    }
    actual = [
        {
            "kind": "invalidation",
            "content": (
                "The user no longer wants to run a marathon."
            ),
            "memory_type": "goal",
            "evidence": (
                "Running a marathon is no longer one of my goals"
            ),
            "confidence": 1.0,
        }
    ]

    assert _score_case(
        case,
        actual,
        schema_valid=True,
    ).passed


def test_semantic_judgment_accepts_faithful_synonym_without_lexical_match() -> None:
    case = {
        "id": "semantic-synonym",
        "tags": ["recurrence"],
        "user_message": (
            "Almost every time a deadline gets close, I wait until the "
            "last evening."
        ),
        "expected": {
            "facts": [
                {
                    "kind": "assertion",
                    "memory_type": "recurring_pattern",
                    "accepted_evidence": [
                        "Almost every time a deadline gets close, I wait "
                        "until the last evening"
                    ],
                    "content_assertions": {
                        "must_include": ["last evening"],
                        "must_not_include": [],
                    },
                }
            ]
        },
    }
    actual = [
        {
            "kind": "assertion",
            "content": (
                "The user repeatedly waits until the final evening when a "
                "deadline approaches."
            ),
            "memory_type": "recurring_pattern",
            "evidence": (
                "Almost every time a deadline gets close, I wait until the "
                "last evening"
            ),
            "confidence": 0.99,
        }
    ]
    judgment = MemoryExtractionSemanticJudgment(
        groups=(
            SemanticFactGroup(
                expected_indices=(0,),
                actual_indices=(0,),
                content_faithful=True,
                atomic_decomposition_valid=True,
                evidence_supports_facts=True,
                reason="Final evening is a faithful synonym.",
            ),
        ),
        unmatched_expected_indices=(),
        unmatched_actual_indices=(),
        reason="The propositions are equivalent.",
    )

    result = _score_case(
        case,
        actual,
        schema_valid=True,
        semantic_judgment=judgment,
    )

    assert result.passed
    assert result.content_matches == 1
    assert result.semantic_groups == [
        {
            "expected_indices": [0],
            "actual_indices": [0],
            "content_faithful": True,
            "atomic_decomposition_valid": True,
            "evidence_supports_facts": True,
            "reason": "Final evening is a faithful synonym.",
        }
    ]


def test_semantic_judgment_cannot_override_deterministic_type_or_evidence() -> None:
    case = _case(expected_facts=[_case()["expected"]["facts"][1]])
    actual = [_actual_facts()[1] | {"memory_type": "goal"}]
    judgment = MemoryExtractionSemanticJudgment(
        groups=(
            SemanticFactGroup(
                expected_indices=(0,),
                actual_indices=(0,),
                content_faithful=True,
                atomic_decomposition_valid=True,
                evidence_supports_facts=True,
                reason="Content meaning matches.",
            ),
        ),
        unmatched_expected_indices=(),
        unmatched_actual_indices=(),
        reason="Matched by content.",
    )

    result = _score_case(
        case,
        actual,
        schema_valid=True,
        semantic_judgment=judgment,
    )

    assert not result.passed
    assert result.content_matches == 1
    assert result.memory_type_matches == 0


def test_semantic_judgment_accepts_valid_one_to_two_atomic_decomposition() -> None:
    case = {
        "id": "atomic-decomposition",
        "tags": ["recurrence", "compound"],
        "user_message": (
            "When I receive difficult messages, I repeatedly rewrite my "
            "reply and then avoid sending it."
        ),
        "expected": {
            "facts": [
                {
                    "kind": "assertion",
                    "memory_type": "recurring_pattern",
                    "accepted_evidence": [
                        "I repeatedly rewrite my reply and then avoid "
                        "sending it"
                    ],
                    "content_assertions": {
                        "must_include": ["rewrite", "avoid", "sending"],
                        "must_not_include": [],
                    },
                }
            ]
        },
    }
    actual = [
        {
            "kind": "assertion",
            "content": "The user repeatedly rewrites difficult replies.",
            "memory_type": "recurring_pattern",
            "evidence": "I repeatedly rewrite my reply",
            "confidence": 0.98,
        },
        {
            "kind": "assertion",
            "content": "The user avoids sending difficult replies.",
            "memory_type": "recurring_pattern",
            "evidence": "and then avoid sending it",
            "confidence": 0.97,
        },
    ]
    judgment = MemoryExtractionSemanticJudgment(
        groups=(
            SemanticFactGroup(
                expected_indices=(0,),
                actual_indices=(0, 1),
                content_faithful=True,
                atomic_decomposition_valid=True,
                evidence_supports_facts=True,
                reason="Both outputs are independently changeable facts.",
            ),
        ),
        unmatched_expected_indices=(),
        unmatched_actual_indices=(),
        reason="One oracle fact has a valid atomic decomposition.",
    )

    result = _score_case(
        case,
        actual,
        schema_valid=True,
        semantic_judgment=judgment,
    )

    assert result.passed
    assert result.strict_matches == 1
    assert result.strict_actual_matches == 2
    assert result.evidence_matches == 1


def test_semantic_evidence_support_accepts_wider_grounded_source_span() -> None:
    case = {
        "id": "wider-evidence",
        "tags": ["evidence"],
        "user_message": (
            "Although it is difficult, honesty matters most to me."
        ),
        "expected": {
            "facts": [
                {
                    "kind": "assertion",
                    "memory_type": "value",
                    "accepted_evidence": ["honesty matters most to me"],
                    "content_assertions": {
                        "must_include": ["honesty"],
                        "must_not_include": [],
                    },
                }
            ]
        },
    }
    actual = [
        {
            "kind": "assertion",
            "content": "The user values honesty above other concerns.",
            "memory_type": "value",
            "evidence": (
                "Although it is difficult, honesty matters most to me."
            ),
            "confidence": 0.99,
        }
    ]
    judgment = MemoryExtractionSemanticJudgment(
        groups=(
            SemanticFactGroup(
                expected_indices=(0,),
                actual_indices=(0,),
                content_faithful=True,
                atomic_decomposition_valid=True,
                evidence_supports_facts=True,
                reason="The wider verbatim span directly supports the fact.",
            ),
        ),
        unmatched_expected_indices=(),
        unmatched_actual_indices=(),
        reason="Evidence is wider than the oracle but remains grounded.",
    )

    result = _score_case(
        case,
        actual,
        schema_valid=True,
        semantic_judgment=judgment,
    )

    assert not _evidence_matches(actual[0]["evidence"], case["expected"]["facts"][0])
    assert result.passed
    assert result.grounded_evidence_count == 1


def test_semantic_judgment_requires_complete_unique_index_accounting() -> None:
    judgment = MemoryExtractionSemanticJudgment(
        groups=(
            SemanticFactGroup(
                expected_indices=(0,),
                actual_indices=(0,),
                content_faithful=True,
                atomic_decomposition_valid=True,
                evidence_supports_facts=True,
                reason="First mapping.",
            ),
            SemanticFactGroup(
                expected_indices=(0,),
                actual_indices=(1,),
                content_faithful=True,
                atomic_decomposition_valid=True,
                evidence_supports_facts=True,
                reason="Duplicate expected mapping.",
            ),
        ),
        unmatched_expected_indices=(),
        unmatched_actual_indices=(),
        reason="Invalid duplicate mapping.",
    )

    with pytest.raises(ValueError, match="duplicate fact indices"):
        _validate_semantic_judgment(
            judgment,
            expected_count=1,
            actual_count=2,
        )


def test_llm_semantic_judge_uses_structured_group_contract(tmp_path) -> None:
    prompt = tmp_path / "judge.txt"
    prompt.write_text("Judge facts semantically.", encoding="utf-8")
    llm = RecordingLLM(
        {
            "groups": [
                {
                    "expected_indices": [0],
                    "actual_indices": [0],
                    "content_faithful": True,
                    "atomic_decomposition_valid": True,
                    "evidence_supports_facts": True,
                    "reason": "Equivalent meaning.",
                }
            ],
            "unmatched_expected_indices": [],
            "unmatched_actual_indices": [],
            "reason": "Complete semantic match.",
        }
    )
    judge = LLMMemoryExtractionJudge(
        llm_client=llm,
        prompt_path=prompt,
        max_attempts=1,
    )

    judgment = asyncio.run(
        judge.judge(
            case={
                "user_message": "I prefer quiet mornings.",
                "expected": {
                    "facts": [
                        {
                            "kind": "assertion",
                            "memory_type": "preference",
                            "accepted_evidence": [
                                "I prefer quiet mornings"
                            ],
                            "content_assertions": {
                                "must_include": ["quiet mornings"],
                                "must_not_include": [],
                            },
                        }
                    ]
                },
            },
            actual_facts=[
                {
                    "kind": "assertion",
                    "content": "The user likes calm mornings.",
                    "memory_type": "preference",
                    "evidence": "I prefer quiet mornings",
                    "confidence": 0.99,
                }
            ],
        )
    )

    assert judgment.groups[0].content_faithful
    assert llm.calls[0][1] == (
        MemoryExtractionSemanticJudgment.model_json_schema()
    )


def test_score_case_detects_missed_fact_and_false_memory() -> None:
    missed = _score_case(
        _case(),
        _actual_facts()[:1],
        schema_valid=True,
    )
    false_memory = _score_case(
        _case(expected_facts=[], tags=["negative"]),
        _actual_facts()[:1],
        schema_valid=True,
    )

    assert not missed.passed
    assert missed.strict_matches == 1
    assert not missed.false_memory
    assert not false_memory.passed
    assert false_memory.false_memory


def test_low_confidence_false_memory_is_reported_but_not_eligible() -> None:
    low_confidence = {
        **_actual_facts()[0],
        "confidence": 0.7,
    }
    result = _score_case(
        _case(expected_facts=[], tags=["negative"]),
        [low_confidence],
        schema_valid=True,
        confidence_threshold=0.85,
    )

    assert result.passed
    assert result.raw_false_memory
    assert not result.false_memory
    assert result.raw_actual_fact_count == 1
    assert result.actual_fact_count == 0
    assert result.low_confidence_fact_count == 1


def test_score_case_requires_expected_evidence_and_grounding() -> None:
    facts = _actual_facts()
    facts[0] = {
        **facts[0],
        "evidence": "invented bakery evidence",
    }
    result = _score_case(
        _case(),
        facts,
        schema_valid=True,
    )

    assert not result.passed
    assert result.evidence_matches == 1
    assert result.grounded_evidence_count == 1


def test_report_computes_precision_recall_and_acceptance_gates() -> None:
    case = _case()
    result = _score_case(
        case,
        _actual_facts(),
        schema_valid=True,
    )
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "LLMMemoryFactExtractor",
        "description": "Fixture.",
        "evaluation_contract": {
            "acceptance_gates": {
                "fact_precision": {"minimum": 0.9},
                "fact_recall": {"minimum": 0.9},
                "false_memory_rate": {"maximum": 0.0},
            }
        },
        "cases": [case],
    }

    report = _build_report(dataset, [result], model="test-model")

    assert report["summary"]["fact_precision"] == 1.0
    assert report["summary"]["fact_recall"] == 1.0
    assert report["summary"]["fact_f1"] == 1.0
    assert report["summary"]["compound_fact_recall"] == 1.0
    assert report["summary"]["false_memory_rate"] == 0.0
    assert report["acceptance_gates"]["passed"]


def test_report_records_candidate_and_judge_models() -> None:
    case = _case()
    result = _score_case(case, _actual_facts(), schema_valid=True)
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "LLMMemoryFactExtractor",
        "description": "Fixture.",
        "evaluation_contract": {"acceptance_gates": {}},
        "cases": [case],
    }

    report = _build_report(
        dataset,
        [result],
        model="gpt-5.4-mini",
        judge_model="gpt-5.6-sol",
    )

    assert report["model"] == "gpt-5.4-mini"
    assert report["judge_model"] == "gpt-5.6-sol"


def test_schema_failure_counts_as_case_failure() -> None:
    case = _case()
    result = ExtractionCaseResult(
        case_id=case["id"],
        passed=False,
        schema_valid=False,
        expected_facts=case["expected"]["facts"],
        actual_facts=[],
        eligible_facts=[],
        strict_matches=0,
        expected_fact_count=2,
        raw_actual_fact_count=0,
        actual_fact_count=0,
        low_confidence_fact_count=0,
        content_matches=0,
        kind_matches=0,
        memory_type_matches=0,
        evidence_matches=0,
        grounded_evidence_count=0,
        duplicate_fact=False,
        raw_false_memory=False,
        false_memory=False,
        error="invalid JSON",
    )
    dataset = {
        "schema_version": "1.0",
        "evaluation_target": "LLMMemoryFactExtractor",
        "description": "Fixture.",
        "evaluation_contract": {"acceptance_gates": {}},
        "cases": [case],
    }

    report = _build_report(dataset, [result])

    assert report["summary"]["schema_valid_response_rate"] == 0.0
    assert report["summary"]["fact_recall"] == 0.0
