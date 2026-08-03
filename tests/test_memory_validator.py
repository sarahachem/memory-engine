import asyncio
import json
from pathlib import Path
from typing import Any

from memory_engine.memory.planner import (
    DeleteMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import Memory
from memory_engine.memory.validator import (
    LLMMemoryMutationValidator,
    MemoryMutationValidation,
)
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository


class RecordingLLMClient:
    def __init__(
        self,
        response: str | None = None,
        error: RuntimeError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.messages: list[Any] | None = None
        self.response_schema: dict[str, Any] | None = None

    async def generate(
        self,
        messages: list[Any],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        self.messages = messages
        self.response_schema = response_schema
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def prompt_repository(tmp_path: Path) -> PromptRepository:
    prompt_dir = tmp_path / "memory"
    prompt_dir.mkdir()
    (prompt_dir / "validate_mutation.txt").write_text(
        "Reject uncertain destructive mutations.",
        encoding="utf-8",
    )
    return PromptRepository(tmp_path)


def target_memory() -> Memory:
    return Memory(
        id="private-id",
        content="The user wants to learn piano.",
        memory_type=MemoryType.GOAL,
        confidence=0.95,
    )


def delete_operation() -> DeleteMemoryOperation:
    return DeleteMemoryOperation(
        operation="delete",
        memory_index=3,
        confidence=0.99,
        explanation="Planner explanation must not reach the validator.",
    )


def test_delete_receives_only_narrow_safety_input(
    tmp_path: Path,
) -> None:
    llm = RecordingLLMClient(
        json.dumps(
            {
                "approved": False,
                "reason": "The message does not invalidate the target.",
            }
        )
    )
    validator = LLMMemoryMutationValidator(
        llm_client=llm,
        prompt_repository=prompt_repository(tmp_path),
    )

    result = asyncio.run(
        validator.validate(
            user_message="Learning guitar is no longer my goal.",
            operation=delete_operation(),
            target_memory=target_memory(),
        )
    )

    assert not result.approved
    assert llm.response_schema == (
        MemoryMutationValidation.model_json_schema()
    )
    assert llm.messages is not None
    payload = json.loads(llm.messages[1].content)
    assert payload == {
        "user_message": "Learning guitar is no longer my goal.",
        "operation": "delete",
        "target_memory": {
            "content": "The user wants to learn piano.",
            "memory_type": "goal",
        },
    }


def test_update_includes_proposed_content_but_no_planner_metadata(
    tmp_path: Path,
) -> None:
    llm = RecordingLLMClient(
        '{"approved": true, "reason": "A supported replacement."}'
    )
    validator = LLMMemoryMutationValidator(
        llm_client=llm,
        prompt_repository=prompt_repository(tmp_path),
    )
    operation = UpdateMemoryOperation(
        operation="update",
        memory_index=7,
        content="The user wants to learn violin.",
        confidence=0.99,
        explanation="Do not pass this through.",
    )

    result = asyncio.run(
        validator.validate(
            user_message="I want violin instead of piano.",
            operation=operation,
            target_memory=target_memory(),
        )
    )

    assert result.approved
    assert llm.messages is not None
    payload = json.loads(llm.messages[1].content)
    assert payload["proposed_content"] == (
        "The user wants to learn violin."
    )
    assert "memory_index" not in payload
    assert "confidence" not in payload
    assert "explanation" not in payload
    assert "private-id" not in llm.messages[1].content


def test_malformed_response_fails_closed(tmp_path: Path) -> None:
    validator = LLMMemoryMutationValidator(
        llm_client=RecordingLLMClient('{"approved": true}'),
        prompt_repository=prompt_repository(tmp_path),
    )

    result = asyncio.run(
        validator.validate(
            user_message="Piano is no longer my goal.",
            operation=delete_operation(),
            target_memory=target_memory(),
        )
    )

    assert not result.approved
    assert "could not be validated safely" in result.reason


def test_llm_failure_fails_closed(tmp_path: Path) -> None:
    validator = LLMMemoryMutationValidator(
        llm_client=RecordingLLMClient(
            error=RuntimeError("validator unavailable")
        ),
        prompt_repository=prompt_repository(tmp_path),
    )

    result = asyncio.run(
        validator.validate(
            user_message="Piano is no longer my goal.",
            operation=delete_operation(),
            target_memory=target_memory(),
        )
    )

    assert not result.approved
