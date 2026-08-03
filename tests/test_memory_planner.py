import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from memory_engine.memory.planner import (
    CreateMemoryOperation,
    DeleteMemoryOperation,
    LLMMemoryPlanner,
    MemoryPlan,
)
from memory_engine.memory.service import Memory
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository


class RecordingLLMClient:
    """
    Deterministic test double that returns one configured response and records
    the request made by LLMMemoryPlanner.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[Any] | None = None
        self.response_schema: dict[str, Any] | None = None

    async def generate(
        self,
        messages: list[Any],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        self.messages = messages
        self.response_schema = response_schema
        return self.response


def make_prompt_repository(tmp_path: Path) -> PromptRepository:
    memory_prompt_directory = tmp_path / "memory"
    memory_prompt_directory.mkdir()
    (memory_prompt_directory / "plan.txt").write_text(
        "Return a valid memory plan.",
        encoding="utf-8",
    )
    return PromptRepository(tmp_path)


def make_memory(
    *,
    memory_id: str = "memory-1",
    content: str = (
        "The user wants to become comfortable speaking in public."
    ),
    memory_type: MemoryType = MemoryType.GOAL,
    confidence: float = 0.95,
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        confidence=confidence,
    )


def test_returns_validated_memory_plan(tmp_path: Path) -> None:
    llm = RecordingLLMClient(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "create",
                        "content": "The user wants to learn Japanese.",
                        "memory_type": "goal",
                        "confidence": 0.96,
                        "explanation": (
                            "The user explicitly stated a durable goal."
                        ),
                    }
                ]
            }
        )
    )
    planner = LLMMemoryPlanner(
        llm_client=llm,
        prompt_repository=make_prompt_repository(tmp_path),
    )

    plan = asyncio.run(
        planner.plan(
            user_message="I want to learn Japanese.",
            active_memories=(),
        )
    )

    assert isinstance(plan, MemoryPlan)
    assert len(plan.operations) == 1
    assert isinstance(plan.operations[0], CreateMemoryOperation)
    assert plan.operations[0].content == (
        "The user wants to learn Japanese."
    )


def test_supplies_memory_plan_schema_to_llm(tmp_path: Path) -> None:
    llm = RecordingLLMClient(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "noop",
                        "confidence": 0.95,
                        "explanation": (
                            "The message contains no durable information."
                        ),
                    }
                ]
            }
        )
    )
    planner = LLMMemoryPlanner(
        llm_client=llm,
        prompt_repository=make_prompt_repository(tmp_path),
    )

    asyncio.run(
        planner.plan(
            user_message="What is the weather like?",
            active_memories=(),
        )
    )

    assert llm.response_schema == MemoryPlan.model_json_schema()


def test_prompt_contains_message_and_temporary_memory_indices(
    tmp_path: Path,
) -> None:
    llm = RecordingLLMClient(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "delete",
                        "memory_index": 1,
                        "confidence": 0.98,
                        "explanation": (
                            "The user explicitly abandoned this goal."
                        ),
                    }
                ]
            }
        )
    )
    planner = LLMMemoryPlanner(
        llm_client=llm,
        prompt_repository=make_prompt_repository(tmp_path),
    )
    memories = (
        make_memory(
            memory_id="private-database-id-1",
            content="The user lives in Munich.",
            memory_type=MemoryType.PERSONAL_FACT,
        ),
        make_memory(
            memory_id="private-database-id-2",
        ),
    )

    plan = asyncio.run(
        planner.plan(
            user_message="Public speaking is no longer my goal.",
            active_memories=memories,
        )
    )

    assert isinstance(plan.operations[0], DeleteMemoryOperation)
    assert llm.messages is not None

    user_prompt = llm.messages[1].content

    assert "Public speaking is no longer my goal." in user_prompt
    assert '"memory_index": 0' in user_prompt
    assert '"memory_index": 1' in user_prompt
    assert "The user lives in Munich." in user_prompt
    assert (
        "The user wants to become comfortable speaking in public."
        in user_prompt
    )
    assert "private-database-id-1" not in user_prompt
    assert "private-database-id-2" not in user_prompt


def test_rejects_malformed_json(tmp_path: Path) -> None:
    llm = RecordingLLMClient("not valid JSON")
    planner = LLMMemoryPlanner(
        llm_client=llm,
        prompt_repository=make_prompt_repository(tmp_path),
    )

    with pytest.raises(
        RuntimeError,
        match="invalid structured response",
    ):
        asyncio.run(
            planner.plan(
                user_message="I want to learn Japanese.",
                active_memories=(),
            )
        )


def test_rejects_schema_invalid_response(tmp_path: Path) -> None:
    llm = RecordingLLMClient(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "create",
                        "memory_type": "goal",
                        "confidence": 0.95,
                        "explanation": "CREATE is missing content.",
                    }
                ]
            }
        )
    )
    planner = LLMMemoryPlanner(
        llm_client=llm,
        prompt_repository=make_prompt_repository(tmp_path),
    )

    with pytest.raises(
        RuntimeError,
        match="invalid structured response",
    ):
        asyncio.run(
            planner.plan(
                user_message="I want to learn Japanese.",
                active_memories=(),
            )
        )


def test_rejects_memory_index_outside_retrieved_set(
    tmp_path: Path,
) -> None:
    llm = RecordingLLMClient(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "delete",
                        "memory_index": 1,
                        "confidence": 0.98,
                        "explanation": (
                            "The model selected a memory that was not supplied."
                        ),
                    }
                ]
            }
        )
    )
    planner = LLMMemoryPlanner(
        llm_client=llm,
        prompt_repository=make_prompt_repository(tmp_path),
    )

    with pytest.raises(
        ValueError,
        match="invalid memory index",
    ):
        asyncio.run(
            planner.plan(
                user_message="Public speaking is no longer my goal.",
                active_memories=(make_memory(),),
            )
        )
