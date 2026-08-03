from __future__ import annotations

import logging

from memory_engine.memory.extractor import MemoryFactExtractor
from memory_engine.memory.extraction_auditor import MemoryExtractionAuditor
from memory_engine.memory.eligibility import MemoryEligibilityGate
from memory_engine.memory.planner import (
    DeleteMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.reconciler import (
    CreateDecision,
    InvalidateDecision,
    MemoryReconciliationRequest,
    MemoryReconciler,
    NoopDecision,
    UpdateDecision,
)
from memory_engine.memory.retriever import MemoryRetriever
from memory_engine.memory.service import Memory, MemoryService
from memory_engine.memory.validator import MemoryMutationValidator
from memory_engine.models import MemoryCandidate

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """
    Production memory capture pipeline.

    The extractor discovers every durable atomic fact in the full message.
    Each fact is then retrieved and reconciled independently. This avoids
    punctuation-based splitting while preventing one clause from suppressing
    another. Storage remains append-only through MemoryService events.
    """

    def __init__(
        self,
        *,
        extractor: MemoryFactExtractor,
        reconciler: MemoryReconciler,
        memory_service: MemoryService,
        memory_retriever: MemoryRetriever,
        mutation_validator: MemoryMutationValidator,
        extraction_auditor: MemoryExtractionAuditor | None = None,
        eligibility_gate: MemoryEligibilityGate | None = None,
        confidence_threshold: float = 0.85,
        retrieval_limit: int = 10,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than 0.")

        self.extractor = extractor
        self.reconciler = reconciler
        self.memory_service = memory_service
        self.memory_retriever = memory_retriever
        self.mutation_validator = mutation_validator
        self.extraction_auditor = extraction_auditor
        self.eligibility_gate = eligibility_gate
        self.confidence_threshold = confidence_threshold
        self.retrieval_limit = retrieval_limit

    async def capture(
        self,
        user_id: str,
        user_message: str,
    ) -> tuple[Memory, ...]:
        if self.eligibility_gate is not None:
            try:
                eligibility = await self.eligibility_gate.assess(user_message)
            except Exception:
                logger.exception(
                    "Memory eligibility assessment failed closed; "
                    "skipping capture for this interaction."
                )
                return ()
            if not eligibility.eligible:
                logger.debug(
                    "Interaction is not eligible for memory capture: %s",
                    eligibility.reason,
                )
                return ()

        episode = await self.memory_service.record_episode(
            user_id=user_id,
            content=user_message,
        )
        fact_set = await self.extractor.extract(user_message)
        if self.extraction_auditor is not None:
            try:
                audit = await self.extraction_auditor.audit(
                    user_message=user_message,
                    candidate_facts=fact_set,
                )
            except Exception:
                logger.exception(
                    "Memory extraction audit failed closed; skipping "
                    "memory capture for this interaction."
                )
                return ()
            if not audit.approved:
                logger.info(
                    "Memory extraction audit corrected %s issue(s).",
                    len(audit.issues),
                )
            fact_set = audit.final_facts
        active_snapshot = await self.memory_service.list_active(
            user_id=user_id,
        )
        reconciliation_requests: list[MemoryReconciliationRequest] = []
        for fact in fact_set.facts:
            if fact.confidence < self.confidence_threshold:
                continue

            try:
                retrieved = await self.memory_retriever.retrieve(
                    query=f"{fact.content} {fact.evidence}",
                    candidate_memories=active_snapshot,
                    limit=self.retrieval_limit,
                )
            except Exception:
                logger.exception(
                    "Memory candidate retrieval failed; skipping "
                    "reconciliation for the current fact."
                )
                continue
            reconciliation_requests.append(
                MemoryReconciliationRequest(
                    fact=fact,
                    candidate_memories=tuple(
                        result.memory for result in retrieved
                    ),
                )
            )

        if not reconciliation_requests:
            return ()

        requests = tuple(reconciliation_requests)
        try:
            decisions = await self.reconciler.reconcile_many(requests)
        except Exception:
            logger.exception(
                "Batched memory reconciliation failed closed; skipping "
                "all mutations for this interaction."
            )
            return ()
        if len(decisions) != len(requests):
            logger.error(
                "Batched memory reconciliation returned %s decisions for "
                "%s facts; skipping all mutations.",
                len(decisions),
                len(requests),
            )
            return ()

        changed: list[Memory] = []
        for request, decision in zip(requests, decisions, strict=True):
            fact = request.fact
            candidates = request.candidate_memories

            if decision.confidence < self.confidence_threshold:
                continue
            if isinstance(decision, NoopDecision):
                continue

            if isinstance(decision, CreateDecision):
                normalized = fact.content.strip().casefold()
                active = await self.memory_service.list_active(user_id)
                if any(
                    memory.content.strip().casefold() == normalized
                    for memory in active
                ):
                    continue
                changed.append(
                    await self.memory_service.save(
                        user_id=user_id,
                        candidate=MemoryCandidate(
                            content=fact.content.strip(),
                            memory_type=fact.memory_type,
                            confidence=min(
                                fact.confidence,
                                decision.confidence,
                            ),
                        ),
                        source_episode_id=episode.id,
                        evidence=fact.evidence,
                    )
                )
                continue

            if decision.memory_index >= len(candidates):
                logger.error(
                    "Skipping %s with invalid candidate index %s for a "
                    "candidate set of size %s.",
                    decision.action,
                    decision.memory_index,
                    len(candidates),
                )
                continue
            target = candidates[decision.memory_index]
            current_target = next(
                (
                    memory
                    for memory in await self.memory_service.list_active(
                        user_id
                    )
                    if memory.id == target.id
                ),
                None,
            )
            if (
                current_target is None
                or current_target.content != target.content
                or current_target.memory_type is not target.memory_type
            ):
                logger.warning(
                    "Skipping %s for stale memory target %s.",
                    decision.action,
                    target.id,
                )
                continue
            target = current_target
            if isinstance(decision, UpdateDecision):
                operation = UpdateMemoryOperation(
                    operation="update",
                    memory_index=decision.memory_index,
                    content=decision.content,
                    confidence=decision.confidence,
                    explanation=decision.explanation,
                )
            else:
                operation = DeleteMemoryOperation(
                    operation="delete",
                    memory_index=decision.memory_index,
                    confidence=decision.confidence,
                    explanation=decision.explanation,
                )

            try:
                validation = await self.mutation_validator.validate(
                    user_message=user_message,
                    operation=operation,
                    target_memory=target,
                )
            except Exception:
                logger.exception(
                    "Memory mutation validation failed closed for %s",
                    decision.action,
                )
                continue

            if not validation.approved:
                logger.warning(
                    "Rejected %s memory operation: %s",
                    decision.action,
                    validation.reason,
                )
                continue

            if isinstance(decision, UpdateDecision):
                changed.append(
                    await self.memory_service.update(
                        user_id=user_id,
                        memory_id=target.id,
                        content=decision.content.strip(),
                        confidence=decision.confidence,
                        source_episode_id=episode.id,
                        evidence=fact.evidence,
                    )
                )
            elif isinstance(decision, InvalidateDecision):
                changed.append(
                    await self.memory_service.invalidate(
                        user_id=user_id,
                        memory_id=target.id,
                        confidence=decision.confidence,
                        source_episode_id=episode.id,
                        evidence=fact.evidence,
                    )
                )

        return tuple(changed)
