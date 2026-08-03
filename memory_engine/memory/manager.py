from __future__ import annotations

import logging

from memory_engine.memory.planner import (
    CreateMemoryOperation,
    DeleteMemoryOperation,
    MemoryPlanner,
    NoopMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import (
    Memory,
    MemoryService,
)
from memory_engine.memory.validator import MemoryMutationValidator
from memory_engine.models import MemoryCandidate

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Legacy one-shot mutation manager retained for baseline evaluations.

    Production dependency wiring uses SemanticMemoryManager.
    """

    def __init__(
        self,
        planner: MemoryPlanner,
        memory_service: MemoryService,
        mutation_validator: MemoryMutationValidator,
        confidence_threshold: float = 0.85,
        retrieval_limit: int = 10,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if retrieval_limit <= 0:
            raise ValueError(
                "retrieval_limit must be greater than 0."
            )

        self.planner = planner
        self.memory_service = memory_service
        self.mutation_validator = mutation_validator
        self.confidence_threshold = confidence_threshold
        self.retrieval_limit = retrieval_limit

    async def capture(
        self,
        user_id: str,
        user_message: str,
    ) -> tuple[Memory, ...]:
        related_memories = await self.memory_service.search(
            user_id=user_id,
            query=user_message,
            limit=self.retrieval_limit,
        )

        logger.debug(
            "Retrieved %d candidate memories for mutation planning",
            len(related_memories),
        )

        plan = await self.planner.plan(
            user_message=user_message,
            active_memories=related_memories,
        )

        active_memories = await self.memory_service.list_active(
            user_id=user_id,
        )

        existing_contents = {
            memory.content.strip().casefold()
            for memory in active_memories
        }

        changed_memories: list[Memory] = []

        for operation in plan.operations:
            if operation.confidence < self.confidence_threshold:
                logger.info(
                    "Skipping low-confidence %s memory operation",
                    operation.operation,
                )
                continue

            if isinstance(
                operation,
                NoopMemoryOperation,
            ):
                continue

            if isinstance(
                operation,
                CreateMemoryOperation,
            ):
                normalized_content = (
                    operation.content.strip().casefold()
                )

                if normalized_content in existing_contents:
                    logger.info(
                        "Skipping exact duplicate memory creation",
                    )
                    continue

                created_memory = await self.memory_service.save(
                    user_id=user_id,
                    candidate=MemoryCandidate(
                        content=operation.content.strip(),
                        memory_type=operation.memory_type,
                        confidence=operation.confidence,
                    ),
                )

                changed_memories.append(
                    created_memory
                )

                existing_contents.add(
                    normalized_content
                )

                continue

            related_memory = related_memories[
                operation.memory_index
            ]

            try:
                validation = await self.mutation_validator.validate(
                    user_message=user_message,
                    operation=operation,
                    target_memory=related_memory,
                )
            except Exception:
                logger.exception(
                    "Memory mutation validation failed; skipping "
                    "%s operation for target index %d",
                    operation.operation,
                    operation.memory_index,
                )
                continue

            if not validation.approved:
                logger.warning(
                    "Rejected %s memory operation: %s",
                    operation.operation,
                    validation.reason,
                )
                continue

            if isinstance(
                operation,
                UpdateMemoryOperation,
            ):
                updated_memory = await self.memory_service.update(
                    user_id=user_id,
                    memory_id=related_memory.id,
                    content=operation.content.strip(),
                    confidence=operation.confidence,
                )

                changed_memories.append(
                    updated_memory
                )

                existing_contents.discard(
                    related_memory.content.strip().casefold()
                )

                existing_contents.add(
                    updated_memory.content.strip().casefold()
                )

                continue

            if isinstance(
                operation,
                DeleteMemoryOperation,
            ):
                deleted_memory = await self.memory_service.delete(
                    user_id=user_id,
                    memory_id=related_memory.id,
                )

                changed_memories.append(
                    deleted_memory
                )

                existing_contents.discard(
                    related_memory.content.strip().casefold()
                )

        return tuple(changed_memories)
