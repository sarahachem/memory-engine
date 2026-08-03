from __future__ import annotations

import argparse
import asyncio
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memory_engine.config import get_settings
from memory_engine.dependencies import (
    get_memory_llm,
    get_prompt_repository,
    get_semantic_llm,
)
from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.memory.extractor import (
    LLMMemoryFactExtractor,
    MemoryFactExtractor,
    MemoryFactSet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "memory_extraction.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
    / "memory_extraction_evaluation.json"
)
DEFAULT_JUDGE_PROMPT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "prompts"
    / "judge_memory_extraction.txt"
)


class SemanticFactGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_indices: tuple[int, ...] = Field(min_length=1)
    actual_indices: tuple[int, ...] = Field(min_length=1)
    content_faithful: bool
    atomic_decomposition_valid: bool
    evidence_supports_facts: bool
    reason: str = Field(min_length=1)


class MemoryExtractionSemanticJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: tuple[SemanticFactGroup, ...]
    unmatched_expected_indices: tuple[int, ...]
    unmatched_actual_indices: tuple[int, ...]
    reason: str = Field(min_length=1)


class MemoryExtractionJudge(ABC):
    @abstractmethod
    async def judge(
        self,
        *,
        case: dict[str, Any],
        actual_facts: list[dict[str, Any]],
    ) -> MemoryExtractionSemanticJudgment:
        raise NotImplementedError


class LLMMemoryExtractionJudge(MemoryExtractionJudge):
    """Offline semantic matcher; never participates in memory capture."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_path: Path = DEFAULT_JUDGE_PROMPT_PATH,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("Judge max_attempts must be positive.")
        self.llm_client = llm_client
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.max_attempts = max_attempts

    async def judge(
        self,
        *,
        case: dict[str, Any],
        actual_facts: list[dict[str, Any]],
    ) -> MemoryExtractionSemanticJudgment:
        expected_facts = case["expected"]["facts"]
        payload = {
            "user_message": case["user_message"],
            "expected_facts": expected_facts,
            "actual_facts": actual_facts,
        }
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.prompt),
            ChatMessage(
                role=ChatRole.USER,
                content=json.dumps(payload, indent=2),
            ),
        ]
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                raw = await self.llm_client.generate(
                    messages=messages,
                    response_schema=(
                        MemoryExtractionSemanticJudgment.model_json_schema()
                    ),
                )
                judgment = (
                    MemoryExtractionSemanticJudgment.model_validate_json(raw)
                )
                _validate_semantic_judgment(
                    judgment,
                    expected_count=len(expected_facts),
                    actual_count=len(actual_facts),
                )
                return judgment
            except (RuntimeError, ValidationError, ValueError) as error:
                last_error = error
        raise RuntimeError(
            f"Memory extraction judge failed after {self.max_attempts} attempts."
        ) from last_error


def _validate_semantic_judgment(
    judgment: MemoryExtractionSemanticJudgment,
    *,
    expected_count: int,
    actual_count: int,
) -> None:
    grouped_expected = [
        index
        for group in judgment.groups
        for index in group.expected_indices
    ]
    grouped_actual = [
        index
        for group in judgment.groups
        for index in group.actual_indices
    ]
    unmatched_expected = list(judgment.unmatched_expected_indices)
    unmatched_actual = list(judgment.unmatched_actual_indices)

    if len(set(grouped_expected)) != len(grouped_expected) or len(
        set(grouped_actual)
    ) != len(grouped_actual):
        raise ValueError("Semantic judge returned duplicate fact indices.")
    if len(set(unmatched_expected)) != len(unmatched_expected) or len(
        set(unmatched_actual)
    ) != len(unmatched_actual):
        raise ValueError("Semantic judge returned duplicate unmatched indices.")
    if set(grouped_expected) & set(unmatched_expected) or set(
        grouped_actual
    ) & set(unmatched_actual):
        raise ValueError("Semantic judge paired and unmatched the same fact.")
    if set(grouped_expected) | set(unmatched_expected) != set(
        range(expected_count)
    ):
        raise ValueError("Semantic judge did not account for every expected fact.")
    if set(grouped_actual) | set(unmatched_actual) != set(range(actual_count)):
        raise ValueError("Semantic judge did not account for every actual fact.")


def _parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLMMemoryFactExtractor against a JSON dataset."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--rescore-from",
        type=Path,
        help=(
            "Recompute scores from an existing report without "
            "calling the model."
        ),
    )
    parser.add_argument(
        "--model",
        help="Explicit OpenAI candidate model used for extraction.",
    )
    parser.add_argument(
        "--judge-model",
        help="Explicit OpenAI model used only for offline semantic scoring.",
    )
    return parser.parse_args(args)


@dataclass(frozen=True)
class ExtractionCaseResult:
    case_id: str
    passed: bool
    schema_valid: bool
    expected_facts: list[dict[str, Any]]
    actual_facts: list[dict[str, Any]]
    eligible_facts: list[dict[str, Any]]
    strict_matches: int
    expected_fact_count: int
    raw_actual_fact_count: int
    actual_fact_count: int
    low_confidence_fact_count: int
    content_matches: int
    kind_matches: int
    memory_type_matches: int
    evidence_matches: int
    grounded_evidence_count: int
    duplicate_fact: bool
    raw_false_memory: bool
    false_memory: bool
    latency_ms: float = 0.0
    semantic_judge_valid: bool = True
    semantic_groups: list[dict[str, Any]] | None = None
    strict_actual_matches: int = 0
    judge_latency_ms: float = 0.0
    judge_reason: str | None = None
    error: str | None = None


def _load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_facts(fact_set: MemoryFactSet) -> list[dict[str, Any]]:
    return [
        fact.model_dump(mode="json")
        for fact in fact_set.facts
    ]


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        normalized = f"{token[:-3]}y"
    else:
        normalized = token

    for suffix in ("ing", "ed", "es", "s"):
        if (
            normalized.endswith(suffix)
            and len(normalized) > len(suffix) + 3
        ):
            normalized = normalized[: -len(suffix)]
            break

    if normalized.endswith("e") and len(normalized) > 4:
        normalized = normalized[:-1]

    return normalized


def _normalized_tokens(text: str) -> list[str]:
    return [
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
    ]


def _contains_concept(content: str, concept: str) -> bool:
    content_tokens = _normalized_tokens(content)
    concept_tokens = _normalized_tokens(concept)

    if not concept_tokens:
        return True

    position = 0
    for token in content_tokens:
        if token == concept_tokens[position]:
            position += 1
            if position == len(concept_tokens):
                return True
    return False


def _content_passes(
    content: str,
    assertions: dict[str, Any],
) -> bool:
    required = all(
        _contains_concept(content, concept)
        for concept in assertions.get("must_include", [])
    )
    alternatives = assertions.get("must_include_any", [])
    alternative = (
        not alternatives
        or any(
            all(
                _contains_concept(content, concept)
                for concept in option
            )
            for option in alternatives
        )
    )
    forbidden = all(
        not _contains_concept(content, concept)
        for concept in assertions.get("must_not_include", [])
    )
    return required and alternative and forbidden


def _accepted_memory_types(expected: dict[str, Any]) -> set[str]:
    return set(
        expected.get(
            "accepted_memory_types",
            [expected["memory_type"]],
        )
    )


def _evidence_matches(
    actual_evidence: str,
    expected: dict[str, Any],
) -> bool:
    def normalize(value: str) -> str:
        return value.strip().casefold().rstrip(".!?").rstrip()

    normalized = normalize(actual_evidence)
    return normalized in {
        normalize(evidence)
        for evidence in expected["accepted_evidence"]
    }


def _is_grounded(evidence: str, user_message: str) -> bool:
    return evidence.strip().casefold() in user_message.casefold()


def _is_duplicate(actual_facts: list[dict[str, Any]]) -> bool:
    keys = [
        (
            fact["kind"],
            fact["memory_type"],
            " ".join(_normalized_tokens(fact["content"])),
        )
        for fact in actual_facts
    ]
    return len(keys) != len(set(keys))


def _score_case(
    case: dict[str, Any],
    actual_facts: list[dict[str, Any]],
    *,
    schema_valid: bool,
    confidence_threshold: float = 0.85,
    latency_ms: float = 0.0,
    semantic_judgment: MemoryExtractionSemanticJudgment | None = None,
    semantic_judge_valid: bool = True,
    judge_latency_ms: float = 0.0,
    error: str | None = None,
) -> ExtractionCaseResult:
    expected_facts = case["expected"]["facts"]
    eligible_facts = [
        fact
        for fact in actual_facts
        if float(fact["confidence"]) >= confidence_threshold
    ]
    strict_matches = 0
    strict_actual_matches = 0
    content_matches = 0
    kind_matches = 0
    memory_type_matches = 0
    evidence_matches = 0

    if semantic_judgment is None:
        unused_actual = set(range(len(eligible_facts)))
        scored_pairs: list[tuple[int, int, bool]] = []
        for expected_index, expected in enumerate(expected_facts):
            candidates = [
                index
                for index in unused_actual
                if _content_passes(
                    eligible_facts[index]["content"],
                    expected["content_assertions"],
                )
            ]
            if not candidates:
                continue

            def candidate_rank(index: int) -> tuple[bool, bool, bool]:
                actual = eligible_facts[index]
                return (
                    actual["kind"] == expected["kind"],
                    actual["memory_type"]
                    in _accepted_memory_types(expected),
                    _evidence_matches(actual["evidence"], expected),
                )

            actual_index = max(candidates, key=candidate_rank)
            unused_actual.remove(actual_index)
            scored_pairs.append((expected_index, actual_index, True))
        for expected_index, actual_index, content_correct in scored_pairs:
            expected = expected_facts[expected_index]
            actual = eligible_facts[actual_index]
            content_matches += int(content_correct)

            kind_correct = actual["kind"] == expected["kind"]
            type_correct = (
                actual["memory_type"] in _accepted_memory_types(expected)
            )
            evidence_correct = _evidence_matches(
                actual["evidence"],
                expected,
            )

            kind_matches += int(kind_correct)
            memory_type_matches += int(type_correct)
            evidence_matches += int(evidence_correct)
            strict = (
                content_correct
                and kind_correct
                and type_correct
                and evidence_correct
            )
            strict_matches += int(strict)
            strict_actual_matches += int(strict)
    else:
        for group in semantic_judgment.groups:
            expected_group = [
                expected_facts[index]
                for index in group.expected_indices
            ]
            actual_group = [
                eligible_facts[index]
                for index in group.actual_indices
            ]
            expected_count = len(expected_group)
            expected_kinds = {
                expected["kind"] for expected in expected_group
            }
            kind_correct = (
                {actual["kind"] for actual in actual_group}
                == expected_kinds
            )
            type_correct = all(
                any(
                    actual["memory_type"]
                    in _accepted_memory_types(expected)
                    for expected in expected_group
                )
                for actual in actual_group
            ) and all(
                any(
                    actual["memory_type"]
                    in _accepted_memory_types(expected)
                    for actual in actual_group
                )
                for expected in expected_group
            )
            content_correct = (
                group.content_faithful
                and group.atomic_decomposition_valid
            )
            evidence_correct = group.evidence_supports_facts

            content_matches += expected_count * int(content_correct)
            kind_matches += expected_count * int(kind_correct)
            memory_type_matches += expected_count * int(type_correct)
            evidence_matches += expected_count * int(evidence_correct)
            strict = (
                content_correct
                and kind_correct
                and type_correct
                and evidence_correct
            )
            strict_matches += expected_count * int(strict)
            strict_actual_matches += len(actual_group) * int(strict)

    grounded_evidence_count = sum(
        _is_grounded(fact["evidence"], case["user_message"])
        for fact in eligible_facts
    )
    duplicate_fact = _is_duplicate(eligible_facts)
    raw_false_memory = not expected_facts and bool(actual_facts)
    false_memory = not expected_facts and bool(eligible_facts)
    passed = (
        schema_valid
        and semantic_judge_valid
        and strict_matches == len(expected_facts)
        and strict_actual_matches == len(eligible_facts)
        and grounded_evidence_count == len(eligible_facts)
        and not duplicate_fact
    )

    return ExtractionCaseResult(
        case_id=case["id"],
        passed=passed,
        schema_valid=schema_valid,
        expected_facts=expected_facts,
        actual_facts=actual_facts,
        eligible_facts=eligible_facts,
        strict_matches=strict_matches,
        expected_fact_count=len(expected_facts),
        raw_actual_fact_count=len(actual_facts),
        actual_fact_count=len(eligible_facts),
        low_confidence_fact_count=(
            len(actual_facts) - len(eligible_facts)
        ),
        content_matches=content_matches,
        kind_matches=kind_matches,
        memory_type_matches=memory_type_matches,
        evidence_matches=evidence_matches,
        grounded_evidence_count=grounded_evidence_count,
        duplicate_fact=duplicate_fact,
        raw_false_memory=raw_false_memory,
        false_memory=false_memory,
        latency_ms=round(latency_ms, 3),
        semantic_judge_valid=semantic_judge_valid,
        semantic_groups=(
            None
            if semantic_judgment is None
            else [
                group.model_dump(mode="json")
                for group in semantic_judgment.groups
            ]
        ),
        strict_actual_matches=strict_actual_matches,
        judge_latency_ms=round(judge_latency_ms, 3),
        judge_reason=(
            None
            if semantic_judgment is None
            else semantic_judgment.reason
        ),
        error=error,
    )


def _safe_rate(
    numerator: int,
    denominator: int,
    *,
    empty_value: float = 1.0,
) -> float:
    if denominator == 0:
        return empty_value
    return round(numerator / denominator, 4)


def _build_report(
    dataset: dict[str, Any],
    results: list[ExtractionCaseResult],
    *,
    model: str | None = None,
    judge_model: str | None = None,
    confidence_threshold: float = 0.85,
) -> dict[str, Any]:
    total_cases = len(results)
    expected_total = sum(
        result.expected_fact_count for result in results
    )
    actual_total = sum(result.actual_fact_count for result in results)
    raw_actual_total = sum(
        result.raw_actual_fact_count for result in results
    )
    strict_expected_total = sum(
        result.strict_matches for result in results
    )
    strict_actual_total = sum(
        result.strict_actual_matches for result in results
    )
    precision = _safe_rate(strict_actual_total, actual_total)
    recall = _safe_rate(strict_expected_total, expected_total)
    f1 = (
        0.0
        if precision + recall == 0
        else round(2 * precision * recall / (precision + recall), 4)
    )

    compound_ids = {
        case["id"]
        for case in dataset["cases"]
        if "compound" in case["tags"]
    }
    compound_results = [
        result
        for result in results
        if result.case_id in compound_ids
    ]
    compound_expected = sum(
        result.expected_fact_count for result in compound_results
    )
    compound_matches = sum(
        result.strict_matches for result in compound_results
    )
    negative_results = [
        result
        for result in results
        if result.expected_fact_count == 0
    ]

    summary = {
        "case_pass_rate": _safe_rate(
            sum(result.passed for result in results),
            total_cases,
        ),
        "schema_valid_response_rate": _safe_rate(
            sum(result.schema_valid for result in results),
            total_cases,
        ),
        "semantic_judge_valid_rate": _safe_rate(
            sum(result.semantic_judge_valid for result in results),
            total_cases,
        ),
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
        "compound_fact_recall": _safe_rate(
            compound_matches,
            compound_expected,
        ),
        "content_faithfulness_rate": _safe_rate(
            sum(result.content_matches for result in results),
            expected_total,
        ),
        "kind_accuracy": _safe_rate(
            sum(result.kind_matches for result in results),
            expected_total,
        ),
        "memory_type_accuracy": _safe_rate(
            sum(result.memory_type_matches for result in results),
            expected_total,
        ),
        "evidence_selection_accuracy": _safe_rate(
            sum(result.evidence_matches for result in results),
            expected_total,
        ),
        "evidence_grounding_rate": _safe_rate(
            sum(
                result.grounded_evidence_count for result in results
            ),
            actual_total,
        ),
        "raw_false_memory_rate": _safe_rate(
            sum(
                result.raw_false_memory
                for result in negative_results
            ),
            len(negative_results),
            empty_value=0.0,
        ),
        "false_memory_rate": _safe_rate(
            sum(result.false_memory for result in negative_results),
            len(negative_results),
            empty_value=0.0,
        ),
        "duplicate_fact_rate": _safe_rate(
            sum(result.duplicate_fact for result in results),
            total_cases,
        ),
        "low_confidence_fact_rate": _safe_rate(
            sum(
                result.low_confidence_fact_count for result in results
            ),
            raw_actual_total,
            empty_value=0.0,
        ),
        "average_latency_ms": round(
            sum(result.latency_ms for result in results) / total_cases,
            3,
        )
        if total_cases
        else 0.0,
        "average_judge_latency_ms": round(
            sum(result.judge_latency_ms for result in results) / total_cases,
            3,
        )
        if total_cases
        else 0.0,
    }

    return {
        "schema_version": dataset["schema_version"],
        "evaluation_target": dataset["evaluation_target"],
        "dataset_description": dataset["description"],
        "model": model,
        "judge_model": judge_model,
        "confidence_threshold": confidence_threshold,
        "summary": summary,
        "acceptance_gates": _evaluate_gates(
            summary,
            dataset["evaluation_contract"].get(
                "acceptance_gates",
                {},
            ),
        ),
        "cases": [asdict(result) for result in results],
    }


def _evaluate_gates(
    summary: dict[str, float],
    gates: dict[str, dict[str, float]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for metric, condition in gates.items():
        value = summary[metric]
        if "minimum" in condition:
            checks[metric] = value >= condition["minimum"]
        elif "maximum" in condition:
            checks[metric] = value <= condition["maximum"]
        else:
            raise ValueError(
                f"Acceptance gate for {metric} has no boundary."
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


async def _evaluate_dataset(
    dataset: dict[str, Any],
    extractor: MemoryFactExtractor,
    *,
    judge: MemoryExtractionJudge | None = None,
    confidence_threshold: float = 0.85,
) -> list[ExtractionCaseResult]:
    results: list[ExtractionCaseResult] = []

    for case in dataset["cases"]:
        started = perf_counter()
        try:
            fact_set = await extractor.extract(case["user_message"])
            latency_ms = (perf_counter() - started) * 1000
            actual_facts = _serialize_facts(fact_set)
            eligible_facts = [
                fact
                for fact in actual_facts
                if float(fact["confidence"]) >= confidence_threshold
            ]
            semantic_judgment = None
            semantic_judge_valid = True
            judge_latency_ms = 0.0
            judge_error = None
            if judge is not None:
                expected_count = len(case["expected"]["facts"])
                actual_count = len(eligible_facts)
                if expected_count == 0 or actual_count == 0:
                    semantic_judgment = MemoryExtractionSemanticJudgment(
                        groups=(),
                        unmatched_expected_indices=tuple(
                            range(expected_count)
                        ),
                        unmatched_actual_indices=tuple(range(actual_count)),
                        reason=(
                            "No semantic pairing is possible because one "
                            "side is empty."
                        ),
                    )
                else:
                    judge_started = perf_counter()
                    try:
                        semantic_judgment = await judge.judge(
                            case=case,
                            actual_facts=eligible_facts,
                        )
                    except Exception as error:
                        semantic_judge_valid = False
                        judge_error = (
                            f"{type(error).__name__}: {error}"
                        )
                        semantic_judgment = (
                            MemoryExtractionSemanticJudgment(
                                groups=(),
                                unmatched_expected_indices=tuple(
                                    range(expected_count)
                                ),
                                unmatched_actual_indices=tuple(
                                    range(actual_count)
                                ),
                                reason="Semantic judge failed closed.",
                            )
                        )
                    judge_latency_ms = (
                        perf_counter() - judge_started
                    ) * 1000
            results.append(
                _score_case(
                    case,
                    actual_facts,
                    schema_valid=True,
                    confidence_threshold=confidence_threshold,
                    latency_ms=latency_ms,
                    semantic_judgment=semantic_judgment,
                    semantic_judge_valid=semantic_judge_valid,
                    judge_latency_ms=judge_latency_ms,
                    error=judge_error,
                )
            )
        except Exception as error:
            latency_ms = (perf_counter() - started) * 1000
            results.append(
                _score_case(
                    case,
                    [],
                    schema_valid=False,
                    confidence_threshold=confidence_threshold,
                    latency_ms=latency_ms,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    return results


async def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    *,
    extractor: MemoryFactExtractor | None = None,
    judge: MemoryExtractionJudge | None = None,
    model: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    settings = get_settings()
    configured_extractor = extractor or LLMMemoryFactExtractor(
        llm_client=get_memory_llm(),
        prompt_repository=get_prompt_repository(),
    )
    configured_judge = judge
    if configured_judge is None and extractor is None:
        configured_judge = LLMMemoryExtractionJudge(
            llm_client=get_semantic_llm(),
        )
    results = await _evaluate_dataset(
        dataset,
        configured_extractor,
        judge=configured_judge,
        confidence_threshold=settings.memory_confidence_threshold,
    )
    report = _build_report(
        dataset,
        results,
        model=(
            model
            if model is not None
            else (
                None
                if extractor is not None
                else (
                    settings.semantic_model
                    if settings.memory_use_semantic_model
                    else settings.memory_model
                )
            )
        ),
        judge_model=(
            judge_model
            if judge_model is not None
            else settings.semantic_model
            if configured_judge is not None
            else None
        ),
        confidence_threshold=settings.memory_confidence_threshold,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def rescore_evaluation(
    dataset_path: Path,
    source_report_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Rescore unchanged model outputs after correcting an oracle."""
    dataset = _load_dataset(dataset_path)
    source_report = json.loads(
        source_report_path.read_text(encoding="utf-8")
    )
    source_cases = {
        case["case_id"]: case
        for case in source_report["cases"]
    }
    dataset_ids = {case["id"] for case in dataset["cases"]}

    if set(source_cases) != dataset_ids:
        missing = sorted(dataset_ids - set(source_cases))
        unexpected = sorted(set(source_cases) - dataset_ids)
        raise ValueError(
            "Dataset and source report case IDs differ. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )

    confidence_threshold = float(
        source_report.get("confidence_threshold", 0.85)
    )

    def source_judgment(
        case: dict[str, Any],
        source_case: dict[str, Any],
    ) -> MemoryExtractionSemanticJudgment | None:
        serialized_groups = source_case.get("semantic_groups")
        if serialized_groups is None:
            # Reports from the retired one-to-one scorer are intentionally
            # rescored through its deterministic compatibility path.
            return None
        groups = tuple(
            SemanticFactGroup.model_validate(group)
            for group in serialized_groups
        )
        grouped_expected = {
            index for group in groups for index in group.expected_indices
        }
        grouped_actual = {
            index for group in groups for index in group.actual_indices
        }
        eligible_count = sum(
            float(fact["confidence"]) >= confidence_threshold
            for fact in source_case.get("actual_facts", [])
        )
        return MemoryExtractionSemanticJudgment(
            groups=groups,
            unmatched_expected_indices=tuple(
                sorted(
                    set(range(len(case["expected"]["facts"])))
                    - grouped_expected
                )
            ),
            unmatched_actual_indices=tuple(
                sorted(set(range(eligible_count)) - grouped_actual)
            ),
            reason=(
                source_case.get("judge_reason")
                or "Preserved semantic judgment from source report."
            ),
        )

    results = [
        _score_case(
            case,
            source_cases[case["id"]].get("actual_facts", []),
            schema_valid=bool(
                source_cases[case["id"]].get("schema_valid", False)
            ),
            confidence_threshold=confidence_threshold,
            latency_ms=float(
                source_cases[case["id"]].get("latency_ms", 0.0)
            ),
            semantic_judgment=source_judgment(
                case,
                source_cases[case["id"]],
            ),
            semantic_judge_valid=bool(
                source_cases[case["id"]].get(
                    "semantic_judge_valid",
                    True,
                )
            ),
            judge_latency_ms=float(
                source_cases[case["id"]].get("judge_latency_ms", 0.0)
            ),
            error=source_cases[case["id"]].get("error"),
        )
        for case in dataset["cases"]
    ]
    report = _build_report(
        dataset,
        results,
        model=source_report.get("model"),
        confidence_threshold=confidence_threshold,
    )
    report["rescore_provenance"] = {
        "model_called": False,
        "source_report": source_report_path.name,
        "source_summary": source_report.get("summary", {}),
        "reason": (
            "Corrected evaluation annotations; model outputs are "
            "unchanged."
        ),
    }
    if "usage" in source_report:
        report["usage"] = source_report["usage"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main(args: list[str] | None = None) -> int:
    from memory_engine.memory.extractor import LLMMemoryFactExtractor
    from evaluation.clients import build_openai_evaluation_client
    from evaluation.usage import (
        attach_named_usage_and_rewrite,
        attach_usage_and_rewrite,
    )

    parsed = _parse_args(args)
    if parsed.rescore_from is None:
        if parsed.model or parsed.judge_model:
            candidate_model = parsed.model or get_settings().semantic_model
            judge_model = (
                parsed.judge_model or get_settings().semantic_model
            )
            candidate_client = build_openai_evaluation_client(
                candidate_model
            )
            judge_client = build_openai_evaluation_client(judge_model)
            report = asyncio.run(
                run_evaluation(
                    parsed.dataset,
                    parsed.report,
                    extractor=LLMMemoryFactExtractor(
                        llm_client=candidate_client,
                        prompt_repository=get_prompt_repository(),
                    ),
                    judge=LLMMemoryExtractionJudge(
                        llm_client=judge_client,
                    ),
                    model=candidate_model,
                    judge_model=judge_model,
                )
            )
            attach_named_usage_and_rewrite(
                report,
                parsed.report,
                {
                    "candidate": candidate_client,
                    "judge": judge_client,
                },
            )
        else:
            report = asyncio.run(
                run_evaluation(parsed.dataset, parsed.report)
            )
            attach_usage_and_rewrite(
                report,
                parsed.report,
                get_memory_llm(),
            )
    else:
        report = rescore_evaluation(
            parsed.dataset,
            parsed.rescore_from,
            parsed.report,
        )
    print(json.dumps(report["summary"], indent=2))
    print(
        "Acceptance gates: "
        + (
            "PASSED"
            if report["acceptance_gates"]["passed"]
            else "FAILED"
        )
    )
    print(f"Report: {parsed.report}")
    return 0 if report["acceptance_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
