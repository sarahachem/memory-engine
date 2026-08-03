from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memory_engine.llm import ChatMessage, ChatRole, LLMClient
from memory_engine.memory.extractor import MemoryFactSet, validate_memory_fact_evidence
from memory_engine.prompting import PromptRepository

logger = logging.getLogger(__name__)


class ExtractionAuditIssueType(StrEnum):
    MISSING_FACT = "missing_fact"
    UNSUPPORTED_FACT = "unsupported_fact"
    NON_ATOMIC_FACT = "non_atomic_fact"
    DUPLICATE_FACT = "duplicate_fact"
    WRONG_KIND = "wrong_kind"
    WRONG_MEMORY_TYPE = "wrong_memory_type"
    UNFAITHFUL_CONTENT = "unfaithful_content"
    INVALID_EVIDENCE = "invalid_evidence"
    INELIGIBLE_FACT = "ineligible_fact"


class ExtractionAuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_type: ExtractionAuditIssueType
    description: str = Field(min_length=1)
    candidate_fact_index: int | None = Field(default=None, ge=0)


class MemoryExtractionAudit(BaseModel):
    """One bounded review whose final facts become authoritative."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    issues: tuple[ExtractionAuditIssue, ...] = ()
    final_facts: MemoryFactSet

    @model_validator(mode="after")
    def validate_verdict(self) -> "MemoryExtractionAudit":
        if self.approved and self.issues:
            raise ValueError("An approved extraction audit cannot contain issues.")
        if not self.approved and not self.issues:
            raise ValueError("A corrected extraction audit must explain an issue.")
        return self


class MemoryExtractionAuditor(ABC):
    @abstractmethod
    async def audit(
        self,
        *,
        user_message: str,
        candidate_facts: MemoryFactSet,
    ) -> MemoryExtractionAudit:
        raise NotImplementedError


class FakeMemoryExtractionAuditor(MemoryExtractionAuditor):
    """Scriptable deterministic test double; never production review."""

    def __init__(
        self,
        *,
        scripted: Sequence[MemoryExtractionAudit] = (),
        default: MemoryExtractionAudit | None = None,
        error: Exception | None = None,
    ) -> None:
        self.scripted = list(scripted)
        self.default = default
        self.error = error
        self.calls: list[tuple[str, MemoryFactSet]] = []

    async def audit(
        self,
        *,
        user_message: str,
        candidate_facts: MemoryFactSet,
    ) -> MemoryExtractionAudit:
        self.calls.append((user_message, candidate_facts))
        if self.error is not None:
            raise self.error
        if self.scripted:
            result = self.scripted.pop(0)
        elif self.default is not None:
            result = self.default
        else:
            raise ValueError("No fake memory extraction audit configured.")
        validate_memory_fact_evidence(user_message, result.final_facts)
        return result


class LLMMemoryExtractionAuditor(MemoryExtractionAuditor):
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_repository: PromptRepository,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_repository = prompt_repository

    async def audit(
        self,
        *,
        user_message: str,
        candidate_facts: MemoryFactSet,
    ) -> MemoryExtractionAudit:
        raw_response = await self.llm_client.generate(
            messages=[
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=self.prompt_repository.load(
                        "memory/audit_extraction"
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=json.dumps(
                        {
                            "user_message": user_message,
                            "candidate_facts": candidate_facts.model_dump(
                                mode="json"
                            )["facts"],
                        },
                        indent=2,
                    ),
                ),
            ],
            response_schema=MemoryExtractionAudit.model_json_schema(),
        )
        logger.debug("Raw memory extraction audit: %s", raw_response)
        try:
            result = MemoryExtractionAudit.model_validate_json(raw_response)
        except ValidationError as error:
            raise RuntimeError(
                "The memory extraction auditor returned an invalid "
                f"structured response. Raw response: {raw_response!r}"
            ) from error
        validate_memory_fact_evidence(user_message, result.final_facts)
        return result
