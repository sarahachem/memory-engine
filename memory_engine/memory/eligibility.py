from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)


class MemoryEligibilityReason(StrEnum):
    POSSIBLE_DURABLE_INFORMATION = "possible_durable_information"
    EXPLICIT_INVALIDATION = "explicit_invalidation"
    NO_DURABLE_INFORMATION = "no_durable_information"
    UNCERTAIN = "uncertain"


class MemoryEligibilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eligible: bool
    reason: MemoryEligibilityReason
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reason(self) -> "MemoryEligibilityAssessment":
        if self.eligible and self.reason is MemoryEligibilityReason.NO_DURABLE_INFORMATION:
            raise ValueError("An eligible interaction needs an eligible reason.")
        if not self.eligible and self.reason is not MemoryEligibilityReason.NO_DURABLE_INFORMATION:
            raise ValueError("Only no_durable_information may be ineligible.")
        return self


class MemoryEligibilityGate(ABC):
    """High-recall semantic gate before episode capture and extraction."""

    @abstractmethod
    async def assess(self, user_message: str) -> MemoryEligibilityAssessment:
        raise NotImplementedError


class FakeMemoryEligibilityGate(MemoryEligibilityGate):
    def __init__(
        self,
        assessments_by_message: Mapping[str, MemoryEligibilityAssessment] | None = None,
        *,
        scripted: Sequence[MemoryEligibilityAssessment] = (),
        default: MemoryEligibilityAssessment | None = None,
        error: Exception | None = None,
    ) -> None:
        self.assessments_by_message = dict(assessments_by_message or {})
        self.scripted = list(scripted)
        self.default = default
        self.error = error
        self.calls: list[str] = []

    async def assess(self, user_message: str) -> MemoryEligibilityAssessment:
        self.calls.append(user_message)
        if self.error is not None:
            raise self.error
        if self.scripted:
            return self.scripted.pop(0)
        if user_message in self.assessments_by_message:
            return self.assessments_by_message[user_message]
        if self.default is not None:
            return self.default
        raise ValueError("No fake memory eligibility assessment configured.")


class LLMMemoryEligibilityGate(MemoryEligibilityGate):
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def assess(self, user_message: str) -> MemoryEligibilityAssessment:
        raw_response = await self.llm_client.generate(
            messages=[
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=self.prompt_repository.load(
                        "memory/assess_eligibility"
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=json.dumps(
                        {"user_message": user_message}, indent=2
                    ),
                ),
            ],
            response_schema=MemoryEligibilityAssessment.model_json_schema(),
        )
        logger.debug("Raw memory eligibility assessment: %s", raw_response)
        try:
            return MemoryEligibilityAssessment.model_validate_json(raw_response)
        except ValidationError as error:
            raise RuntimeError(
                "The memory eligibility model returned an invalid "
                f"structured response. Raw response: {raw_response!r}"
            ) from error
