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

from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.models import MemoryType
from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)


class AssertionMemoryFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["assertion"]
    content: str = Field(min_length=1)
    memory_type: MemoryType
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class InvalidationMemoryFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["invalidation"]
    content: str = Field(min_length=1)
    memory_type: MemoryType
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


MemoryFact = Annotated[
    AssertionMemoryFact | InvalidationMemoryFact,
    Field(discriminator="kind"),
]


class StructuredMemoryFact(BaseModel):
    """Provider-compatible envelope converted into the domain union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["assertion", "invalidation"]
    content: str = Field(min_length=1)
    memory_type: MemoryType
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    def to_domain_fact(self) -> AssertionMemoryFact | InvalidationMemoryFact:
        values = self.model_dump(mode="python")
        if self.kind == "assertion":
            return AssertionMemoryFact.model_validate(values)
        return InvalidationMemoryFact.model_validate(values)


class StructuredMemoryFactSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: tuple[StructuredMemoryFact, ...] = ()


class MemoryFactSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: tuple[MemoryFact, ...] = ()

    @model_validator(mode="after")
    def collapse_exact_duplicates(self) -> "MemoryFactSet":
        """Collapse structurally identical facts with different evidence."""
        unique_facts: list[MemoryFact] = []
        seen: set[tuple[str, MemoryType, str]] = set()

        for fact in self.facts:
            key = (
                fact.kind,
                fact.memory_type,
                " ".join(fact.content.split()).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_facts.append(fact)

        self.facts = tuple(unique_facts)
        return self


def validate_memory_fact_evidence(
    user_message: str,
    fact_set: MemoryFactSet,
) -> None:
    """Reject model-produced evidence not found in the source message."""
    message_normalized = user_message.casefold()
    for fact in fact_set.facts:
        if fact.evidence.strip().casefold() not in message_normalized:
            raise RuntimeError(
                "The memory model returned evidence that is not present "
                "in the user message."
            )


class MemoryFactExtractor(ABC):
    @abstractmethod
    async def extract(self, user_message: str) -> MemoryFactSet:
        raise NotImplementedError


class LLMMemoryFactExtractor(MemoryFactExtractor):
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def extract(self, user_message: str) -> MemoryFactSet:
        raw_response = await self.llm_client.generate(
            messages=[
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=self.prompt_repository.load(
                        "memory/extract_facts"
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=json.dumps(
                        {"user_message": user_message},
                        indent=2,
                    ),
                ),
            ],
            response_schema=StructuredMemoryFactSet.model_json_schema(),
        )

        logger.debug("Raw memory fact response: %s", raw_response)

        try:
            provider_facts = StructuredMemoryFactSet.model_validate_json(
                raw_response
            )
            facts = MemoryFactSet(
                facts=tuple(
                    fact.to_domain_fact() for fact in provider_facts.facts
                )
            )
        except ValidationError as error:
            raise RuntimeError(
                "The memory extraction model returned an invalid "
                f"structured response. Raw response: {raw_response!r}"
            ) from error

        validate_memory_fact_evidence(user_message, facts)

        return facts
