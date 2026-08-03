from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.memory.planner import (
    DeleteMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import Memory
from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)

DestructiveMemoryOperation = (
    UpdateMemoryOperation | DeleteMemoryOperation
)


class MemoryMutationValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: str = Field(min_length=1)


class MemoryMutationValidator(ABC):
    @abstractmethod
    async def validate(
        self,
        *,
        user_message: str,
        operation: DestructiveMemoryOperation,
        target_memory: Memory,
    ) -> MemoryMutationValidation:
        raise NotImplementedError


class LLMMemoryMutationValidator(MemoryMutationValidator):
    """
    Fail-closed validation boundary for proposed UPDATE and DELETE operations.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def validate(
        self,
        *,
        user_message: str,
        operation: DestructiveMemoryOperation,
        target_memory: Memory,
    ) -> MemoryMutationValidation:
        messages = self._build_messages(
            user_message=user_message,
            operation=operation,
            target_memory=target_memory,
        )

        try:
            raw_response = await self.llm_client.generate(
                messages=messages,
                response_schema=(
                    MemoryMutationValidation.model_json_schema()
                ),
            )
            return MemoryMutationValidation.model_validate_json(
                raw_response
            )
        except (RuntimeError, ValidationError) as error:
            logger.warning(
                "Memory mutation validation failed closed: %s",
                error,
            )
            return MemoryMutationValidation(
                approved=False,
                reason=(
                    "The destructive operation could not be validated "
                    "safely."
                ),
            )

    def _build_messages(
        self,
        *,
        user_message: str,
        operation: DestructiveMemoryOperation,
        target_memory: Memory,
    ) -> list[ChatMessage]:
        payload: dict[str, object] = {
            "user_message": user_message,
            "operation": operation.operation,
            "target_memory": {
                "content": target_memory.content,
                "memory_type": target_memory.memory_type.value,
            },
        }

        if isinstance(operation, UpdateMemoryOperation):
            payload["proposed_content"] = operation.content

        return [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=self.prompt_repository.load(
                    "memory/validate_mutation"
                ),
            ),
            ChatMessage(
                role=ChatRole.USER,
                content=json.dumps(payload, indent=2),
            ),
        ]
