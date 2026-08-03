from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from memory_engine.llm import (
    ChatMessage,
    ChatRole,
    LLMClient,
)
from memory_engine.memory.service import Memory
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)


class CreateMemoryOperation(BaseModel):
    """
    Creates a new long-term memory.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    operation: Literal["create"]

    content: str = Field(
        min_length=1,
    )

    memory_type: MemoryType

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    explanation: str = Field(
        min_length=1,
    )


class UpdateMemoryOperation(BaseModel):
    """
    Updates one retrieved memory.

    memory_index refers to the temporary position of the memory
    inside the retrieved memories supplied to the planner.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    operation: Literal["update"]

    memory_index: int = Field(
        ge=0,
    )

    content: str = Field(
        min_length=1,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    explanation: str = Field(
        min_length=1,
    )


class DeleteMemoryOperation(BaseModel):
    """
    Deletes one retrieved memory.

    memory_index refers to the temporary position of the memory
    inside the retrieved memories supplied to the planner.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    operation: Literal["delete"]

    memory_index: int = Field(
        ge=0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    explanation: str = Field(
        min_length=1,
    )


class NoopMemoryOperation(BaseModel):
    """
    Indicates that no long-term memory change is required.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    operation: Literal["noop"]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    explanation: str = Field(
        min_length=1,
    )


MemoryOperation = Annotated[
    (
        CreateMemoryOperation
        | UpdateMemoryOperation
        | DeleteMemoryOperation
        | NoopMemoryOperation
    ),
    Field(
        discriminator="operation",
    ),
]


class MemoryPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    operations: tuple[MemoryOperation, ...]

    @model_validator(mode="after")
    def validate_operations(
        self,
    ) -> "MemoryPlan":
        if not self.operations:
            raise ValueError(
                "A memory plan must contain at least one operation."
            )

        noop_operations = [
            operation
            for operation in self.operations
            if isinstance(
                operation,
                NoopMemoryOperation,
            )
        ]

        if noop_operations and len(self.operations) > 1:
            raise ValueError(
                "NOOP cannot be combined with other operations."
            )

        referenced_indices: set[int] = set()

        for operation in self.operations:
            if not isinstance(
                operation,
                (
                    UpdateMemoryOperation,
                    DeleteMemoryOperation,
                ),
            ):
                continue

            if operation.memory_index in referenced_indices:
                raise ValueError(
                    "A memory plan cannot modify the same existing "
                    "memory more than once."
                )

            referenced_indices.add(
                operation.memory_index
            )

        return self


class MemoryPlanner(ABC):
    @abstractmethod
    async def plan(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> MemoryPlan:
        raise NotImplementedError


class LLMMemoryPlanner(MemoryPlanner):
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def plan(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> MemoryPlan:
        messages = self._build_messages(
            user_message=user_message,
            active_memories=active_memories,
        )

        raw_response = await self.llm_client.generate(
            messages=messages,
            response_schema=MemoryPlan.model_json_schema(),
        )

        logger.debug("Raw memory plan response: %s", raw_response)

        try:
            plan = MemoryPlan.model_validate_json(
                raw_response
            )
        except ValidationError as error:
            raise RuntimeError(
                "The memory planning model returned an invalid "
                "structured response. "
                f"Raw response: {raw_response!r}"
            ) from error

        self._validate_memory_indices(
            plan=plan,
            active_memories=active_memories,
        )

        return plan

    def _build_messages(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> list[ChatMessage]:
        memory_instructions = self.prompt_repository.load(
            "memory/plan"
        )

        memory_input = self._build_input(
            user_message=user_message,
            active_memories=active_memories,
        )

        return [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=memory_instructions,
            ),
            ChatMessage(
                role=ChatRole.USER,
                content=memory_input,
            ),
        ]

    @staticmethod
    def _build_input(
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> str:
        memories_payload = [
            {
                "memory_index": index,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "confidence": memory.confidence,
            }
            for index, memory in enumerate(
                active_memories
            )
        ]

        return (
            "Determine all long-term memory operations required by "
            "the user's original message.\n\n"
            "USER MESSAGE:\n"
            f"{user_message}\n\n"
            "ACTIVE MEMORIES:\n"
            f"{json.dumps(memories_payload, indent=2)}\n\n"
            "A single user message may require multiple operations. "
            "Return only an object matching the required JSON schema."
        )

    @staticmethod
    def _validate_memory_indices(
        plan: MemoryPlan,
        active_memories: tuple[Memory, ...],
    ) -> None:
        memory_count = len(active_memories)

        for operation in plan.operations:
            if not isinstance(
                operation,
                (
                    UpdateMemoryOperation,
                    DeleteMemoryOperation,
                ),
            ):
                continue

            if operation.memory_index >= memory_count:
                raise ValueError(
                    "Memory planner referenced an invalid memory "
                    f"index: {operation.memory_index}. "
                    f"Allowed indices: "
                    f"{list(range(memory_count))}."
                )
