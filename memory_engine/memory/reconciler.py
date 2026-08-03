from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.memory.extractor import MemoryFact
from memory_engine.memory.service import Memory
from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)


class CreateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["create"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


class UpdateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["update"]
    memory_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


class InvalidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["invalidate"]
    memory_index: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


class NoopDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["noop"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


MemoryReconciliationDecision = Annotated[
    CreateDecision | UpdateDecision | InvalidateDecision | NoopDecision,
    Field(discriminator="action"),
]


class StructuredMemoryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_index: int = Field(ge=0)
    action: Literal["create", "update", "invalidate", "noop"]
    memory_index: int | None = Field(default=None, ge=0)
    content: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "StructuredMemoryDecision":
        if self.action == "update":
            if self.memory_index is None or not (self.content or "").strip():
                raise ValueError("UPDATE requires memory_index and content.")
        elif self.action == "invalidate":
            if self.memory_index is None:
                raise ValueError("INVALIDATE requires memory_index.")
            self.content = None
        else:
            # OpenAI strict schemas require nullable fields to be present.
            # Irrelevant values cannot affect CREATE/NOOP execution, so clear
            # them at the transport boundary instead of rejecting an otherwise
            # complete batch. Required destructive fields remain strict above.
            self.memory_index = None
            self.content = None
        return self

    def to_domain_decision(self) -> MemoryReconciliationDecision:
        common = {
            "confidence": self.confidence,
            "explanation": self.explanation,
        }
        if self.action == "create":
            return CreateDecision(action="create", **common)
        if self.action == "noop":
            return NoopDecision(action="noop", **common)
        if self.action == "invalidate":
            return InvalidateDecision(
                action="invalidate",
                memory_index=self.memory_index,
                **common,
            )
        return UpdateDecision(
            action="update",
            memory_index=self.memory_index,
            content=self.content,
            **common,
        )


class MemoryDecisionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: tuple[StructuredMemoryDecision, ...]


@dataclass(frozen=True)
class MemoryReconciliationRequest:
    fact: MemoryFact
    candidate_memories: tuple[Memory, ...]


class MemoryReconciler(ABC):
    @abstractmethod
    async def reconcile(
        self,
        fact: MemoryFact,
        candidate_memories: tuple[Memory, ...],
    ) -> MemoryReconciliationDecision:
        raise NotImplementedError

    async def reconcile_many(
        self,
        requests: tuple[MemoryReconciliationRequest, ...],
    ) -> tuple[MemoryReconciliationDecision, ...]:
        """Compatibility fallback for deterministic and legacy reconcilers."""
        return tuple(
            [
                await self.reconcile(
                    request.fact,
                    request.candidate_memories,
                )
                for request in requests
            ]
        )


class LLMMemoryReconciler(MemoryReconciler):
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def reconcile(
        self,
        fact: MemoryFact,
        candidate_memories: tuple[Memory, ...],
    ) -> MemoryReconciliationDecision:
        decisions = await self.reconcile_many(
            (
                MemoryReconciliationRequest(
                    fact=fact,
                    candidate_memories=candidate_memories,
                ),
            )
        )
        return decisions[0]

    async def reconcile_many(
        self,
        requests: tuple[MemoryReconciliationRequest, ...],
    ) -> tuple[MemoryReconciliationDecision, ...]:
        if not requests:
            return ()

        payload = {
            "facts": [
                {
                    "fact_index": fact_index,
                    "fact": request.fact.model_dump(mode="json"),
                    "candidate_memories": [
                        {
                            "memory_index": memory_index,
                            "candidate_id": memory.id,
                            "content": memory.content,
                            "memory_type": memory.memory_type.value,
                            "confidence": memory.confidence,
                        }
                        for memory_index, memory in enumerate(
                            request.candidate_memories
                        )
                    ],
                }
                for fact_index, request in enumerate(requests)
            ],
        }
        raw_response = await self.llm_client.generate(
            messages=[
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=self.prompt_repository.load(
                        "memory/reconcile_facts"
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=json.dumps(payload, indent=2),
                ),
            ],
            response_schema=MemoryDecisionBatch.model_json_schema(),
        )

        logger.debug(
            "Raw batched memory reconciliation response: %s",
            raw_response,
        )

        try:
            batch = MemoryDecisionBatch.model_validate_json(
                raw_response
            )
        except ValidationError as error:
            raise RuntimeError(
                "The batched memory reconciliation model returned an invalid "
                f"structured response. Raw response: {raw_response!r}"
            ) from error

        expected_indices = set(range(len(requests)))
        returned_indices = [item.fact_index for item in batch.results]
        if (
            len(returned_indices) != len(requests)
            or len(set(returned_indices)) != len(returned_indices)
            or set(returned_indices) != expected_indices
        ):
            raise RuntimeError(
                "The batched memory reconciler must return exactly one "
                "decision for every input fact."
            )

        decisions_by_index = {
            item.fact_index: item.to_domain_decision()
            for item in batch.results
        }
        decisions: list[MemoryReconciliationDecision] = []
        for fact_index, request in enumerate(requests):
            decision = decisions_by_index[fact_index]
            if isinstance(decision, (UpdateDecision, InvalidateDecision)) and (
                decision.memory_index >= len(request.candidate_memories)
            ):
                raise ValueError(
                    "Memory reconciler referenced an invalid memory index "
                    f"for fact {fact_index}: {decision.memory_index}."
                )

            if request.fact.kind == "invalidation" and isinstance(
                decision, (CreateDecision, UpdateDecision)
            ):
                raise ValueError(
                    "An invalidation fact cannot create or update a memory."
                )
            decisions.append(decision)

        return tuple(decisions)
