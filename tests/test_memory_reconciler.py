import asyncio
import json

import pytest

from memory_engine.memory.extractor import (
    AssertionMemoryFact,
    InvalidationMemoryFact,
)
from memory_engine.memory.reconciler import (
    LLMMemoryReconciler,
    MemoryReconciliationRequest,
    NoopDecision,
    UpdateDecision,
)
from memory_engine.memory.service import Memory
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository


class RecordingLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []

    async def generate(self, messages, response_schema=None):
        self.calls.append((messages, response_schema))
        return json.dumps(self.response)


def _repository(tmp_path) -> PromptRepository:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "reconcile_facts.txt").write_text(
        "Reconcile facts.",
        encoding="utf-8",
    )
    return PromptRepository(tmp_path)


def _assertion(content: str = "The user prefers remote work."):
    return AssertionMemoryFact(
        kind="assertion",
        content=content,
        memory_type=MemoryType.PREFERENCE,
        evidence="I prefer remote work",
        confidence=0.98,
    )


def _invalidation():
    return InvalidationMemoryFact(
        kind="invalidation",
        content="The user no longer wants to open a bakery.",
        memory_type=MemoryType.GOAL,
        evidence="I no longer want to open a bakery",
        confidence=0.99,
    )


def _memory(memory_id: str, content: str, memory_type=MemoryType.PREFERENCE):
    return Memory(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        confidence=0.95,
    )


def test_llm_reconciles_multiple_facts_in_one_call_and_restores_order(
    tmp_path,
) -> None:
    llm = RecordingLLM(
        {
            "results": [
                {
                    "fact_index": 1,
                    "action": "noop",
                    "memory_index": 0,
                    "content": None,
                    "confidence": 0.99,
                    "explanation": "No matching goal.",
                },
                {
                    "fact_index": 0,
                    "action": "update",
                    "memory_index": 0,
                    "content": "The user strongly prefers remote work.",
                    "confidence": 0.96,
                    "explanation": "Same preference with added detail.",
                },
            ]
        }
    )
    reconciler = LLMMemoryReconciler(llm, _repository(tmp_path))
    requests = (
        MemoryReconciliationRequest(
            fact=_assertion(),
            candidate_memories=(
                _memory("preference-1", "The user prefers remote work."),
            ),
        ),
        MemoryReconciliationRequest(
            fact=_invalidation(),
            candidate_memories=(),
        ),
    )

    decisions = asyncio.run(reconciler.reconcile_many(requests))

    assert len(llm.calls) == 1
    assert isinstance(decisions[0], UpdateDecision)
    assert isinstance(decisions[1], NoopDecision)
    payload = json.loads(llm.calls[0][0][1].content)
    assert [item["fact_index"] for item in payload["facts"]] == [0, 1]
    assert payload["facts"][0]["candidate_memories"][0][
        "memory_index"
    ] == 0
    assert payload["facts"][0]["candidate_memories"][0][
        "candidate_id"
    ] == "preference-1"


@pytest.mark.parametrize(
    "results",
    [
        [],
        [
            {
                "fact_index": 1,
                "action": "noop",
                "memory_index": None,
                "content": None,
                "confidence": 0.9,
                "explanation": "Unknown fact.",
            }
        ],
        [
            {
                "fact_index": 0,
                "action": "noop",
                "memory_index": None,
                "content": None,
                "confidence": 0.9,
                "explanation": "First.",
            },
            {
                "fact_index": 0,
                "action": "noop",
                "memory_index": None,
                "content": None,
                "confidence": 0.9,
                "explanation": "Duplicate.",
            },
        ],
    ],
)
def test_batch_rejects_missing_unknown_or_duplicate_fact_indices(
    tmp_path,
    results,
) -> None:
    reconciler = LLMMemoryReconciler(
        RecordingLLM({"results": results}),
        _repository(tmp_path),
    )

    with pytest.raises(RuntimeError, match="exactly one decision"):
        asyncio.run(
            reconciler.reconcile_many(
                (
                    MemoryReconciliationRequest(
                        fact=_assertion(),
                        candidate_memories=(),
                    ),
                )
            )
        )


def test_batch_rejects_candidate_index_outside_its_fact_scope(tmp_path) -> None:
    reconciler = LLMMemoryReconciler(
        RecordingLLM(
            {
                "results": [
                    {
                        "fact_index": 0,
                        "action": "update",
                        "memory_index": 1,
                        "content": "Updated.",
                        "confidence": 0.9,
                        "explanation": "Wrong local index.",
                    }
                ]
            }
        ),
        _repository(tmp_path),
    )

    with pytest.raises(ValueError, match="fact 0"):
        asyncio.run(
            reconciler.reconcile_many(
                (
                    MemoryReconciliationRequest(
                        fact=_assertion(),
                        candidate_memories=(
                            _memory("only-candidate", "Original."),
                        ),
                    ),
                )
            )
        )


def test_batch_rejects_create_for_invalidation_fact(tmp_path) -> None:
    reconciler = LLMMemoryReconciler(
        RecordingLLM(
            {
                "results": [
                    {
                        "fact_index": 0,
                        "action": "create",
                        "memory_index": None,
                        "content": None,
                        "confidence": 0.9,
                        "explanation": "Invalid action.",
                    }
                ]
            }
        ),
        _repository(tmp_path),
    )

    with pytest.raises(ValueError, match="cannot create"):
        asyncio.run(
            reconciler.reconcile_many(
                (
                    MemoryReconciliationRequest(
                        fact=_invalidation(),
                        candidate_memories=(),
                    ),
                )
            )
        )


def test_batch_discards_fields_irrelevant_to_create_and_noop(tmp_path) -> None:
    reconciler = LLMMemoryReconciler(
        RecordingLLM(
            {
                "results": [
                    {
                        "fact_index": 0,
                        "action": "create",
                        "memory_index": 9,
                        "content": "Irrelevant generated content.",
                        "confidence": 0.9,
                        "explanation": "New fact.",
                    }
                ]
            }
        ),
        _repository(tmp_path),
    )

    decision = asyncio.run(
        reconciler.reconcile_many(
            (
                MemoryReconciliationRequest(
                    fact=_assertion(),
                    candidate_memories=(),
                ),
            )
        )
    )[0]

    assert decision.action == "create"
    assert not hasattr(decision, "memory_index")
    assert not hasattr(decision, "content")
