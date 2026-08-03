import asyncio
import logging

import pytest

from memory_engine.memory.manager import MemoryManager
from memory_engine.memory.planner import (
    CreateMemoryOperation,
    DeleteMemoryOperation,
    MemoryPlan,
    MemoryPlanner,
    NoopMemoryOperation,
    UpdateMemoryOperation,
)
from memory_engine.memory.service import (
    InMemoryMemoryService,
    Memory,
    MemoryStatus,
)
from memory_engine.memory.validator import (
    MemoryMutationValidation,
    MemoryMutationValidator,
)
from memory_engine.models import MemoryType


USER_ID = "user-1"


class ApprovingMutationValidator(MemoryMutationValidator):
    async def validate(
        self,
        *,
        user_message,
        operation,
        target_memory,
    ) -> MemoryMutationValidation:
        return MemoryMutationValidation(
            approved=True,
            reason="Approved by the deterministic test validator.",
        )


class RejectingMutationValidator(MemoryMutationValidator):
    def __init__(self) -> None:
        self.calls = []

    async def validate(
        self,
        *,
        user_message,
        operation,
        target_memory,
    ) -> MemoryMutationValidation:
        self.calls.append(
            (user_message, operation, target_memory)
        )
        return MemoryMutationValidation(
            approved=False,
            reason="The operation does not safely match the target.",
        )


class ScriptedMutationValidator(MemoryMutationValidator):
    def __init__(
        self,
        outcomes: list[bool | Exception],
    ) -> None:
        self.outcomes = outcomes
        self.calls = []

    async def validate(
        self,
        *,
        user_message,
        operation,
        target_memory,
    ) -> MemoryMutationValidation:
        self.calls.append(
            (user_message, operation, target_memory)
        )
        outcome = self.outcomes[len(self.calls) - 1]

        if isinstance(outcome, Exception):
            raise outcome

        return MemoryMutationValidation(
            approved=outcome,
            reason=(
                "Approved by scripted validator."
                if outcome
                else "Rejected by scripted validator."
            ),
        )


class StaticMemoryPlanner(MemoryPlanner):
    """Returns a predetermined valid plan and records its input."""

    def __init__(self, plan: MemoryPlan) -> None:
        self.plan_to_return = plan
        self.user_message: str | None = None
        self.active_memories: tuple[Memory, ...] | None = None

    async def plan(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> MemoryPlan:
        self.user_message = user_message
        self.active_memories = active_memories
        return self.plan_to_return


def make_memory(
    *,
    memory_id: str,
    content: str,
    memory_type: MemoryType,
    confidence: float = 0.95,
) -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        memory_type=memory_type,
        confidence=confidence,
    )


def make_service(
    *memories: Memory,
) -> InMemoryMemoryService:
    return InMemoryMemoryService(
        memories_by_user={
            USER_ID: tuple(memories),
        }
    )


def run_capture(
    manager: MemoryManager,
    message: str,
) -> tuple[Memory, ...]:
    return asyncio.run(
        manager.capture(
            user_id=USER_ID,
            user_message=message,
        )
    )


def test_create_saves_new_memory() -> None:
    service = make_service()
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                CreateMemoryOperation(
                    operation="create",
                    content="The user wants to learn Japanese.",
                    memory_type=MemoryType.GOAL,
                    confidence=0.96,
                    explanation=(
                        "The user explicitly stated a durable goal."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )

    changed = run_capture(
        manager,
        "I want to learn Japanese.",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert len(changed) == 1
    assert changed[0].content == "The user wants to learn Japanese."
    assert changed[0].memory_type is MemoryType.GOAL
    assert active == changed


def test_create_does_not_call_mutation_validator() -> None:
    service = make_service()
    validator = ScriptedMutationValidator([])
    manager = MemoryManager(
        planner=StaticMemoryPlanner(
            MemoryPlan(
                operations=(
                    CreateMemoryOperation(
                        operation="create",
                        content="The user wants to learn Japanese.",
                        memory_type=MemoryType.GOAL,
                        confidence=0.96,
                        explanation="A durable new goal.",
                    ),
                )
            )
        ),
        memory_service=service,
        mutation_validator=validator,
    )

    changed = run_capture(
        manager,
        "I want to learn Japanese.",
    )

    assert len(changed) == 1
    assert validator.calls == []


def test_exact_duplicate_create_is_skipped() -> None:
    existing = make_memory(
        memory_id="goal-1",
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(existing)
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                CreateMemoryOperation(
                    operation="create",
                    content="  THE USER WANTS TO LEARN JAPANESE.  ",
                    memory_type=MemoryType.GOAL,
                    confidence=0.96,
                    explanation=(
                        "The model proposed an exact duplicate."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )

    changed = run_capture(
        manager,
        "I want to learn Japanese.",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert changed == ()
    assert active == (existing,)


def test_update_targets_retrieved_memory_by_temporary_index() -> None:
    unrelated = make_memory(
        memory_id="preference-1",
        content="The user prefers concise explanations.",
        memory_type=MemoryType.PREFERENCE,
    )
    old_goal = make_memory(
        memory_id="goal-1",
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(unrelated, old_goal)
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                UpdateMemoryOperation(
                    operation="update",
                    memory_index=0,
                    content="The user wants to learn Korean.",
                    confidence=0.97,
                    explanation=(
                        "The user changed the language they want to learn."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )

    changed = run_capture(
        manager,
        "I want to learn Korean instead of Japanese.",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert planner.active_memories == (old_goal,)
    assert len(changed) == 1
    assert changed[0].id == "goal-1"
    assert changed[0].content == "The user wants to learn Korean."
    assert changed[0].memory_type is MemoryType.GOAL
    assert active == (unrelated, changed[0])


def test_delete_targets_retrieved_memory_by_temporary_index() -> None:
    public_speaking_goal = make_memory(
        memory_id="goal-1",
        content=(
            "The user wants to become comfortable speaking in public."
        ),
        memory_type=MemoryType.GOAL,
    )
    service = make_service(public_speaking_goal)
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                DeleteMemoryOperation(
                    operation="delete",
                    memory_index=0,
                    confidence=0.99,
                    explanation=(
                        "The user explicitly abandoned the existing goal."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )

    changed = run_capture(
        manager,
        "Public speaking is no longer my goal.",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert planner.active_memories == (public_speaking_goal,)
    assert len(changed) == 1
    assert changed[0].id == "goal-1"
    assert changed[0].status is MemoryStatus.DELETED
    assert active == ()


def test_delete_and_create_are_both_applied() -> None:
    old_goal = make_memory(
        memory_id="goal-1",
        content=(
            "The user wants to become comfortable speaking in public."
        ),
        memory_type=MemoryType.GOAL,
    )
    service = make_service(old_goal)
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                DeleteMemoryOperation(
                    operation="delete",
                    memory_index=0,
                    confidence=0.99,
                    explanation=(
                        "The user explicitly abandoned the existing goal."
                    ),
                ),
                CreateMemoryOperation(
                    operation="create",
                    content=(
                        "The user wants to become a better leader."
                    ),
                    memory_type=MemoryType.GOAL,
                    confidence=0.97,
                    explanation=(
                        "The user explicitly stated a new durable goal."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )

    changed = run_capture(
        manager,
        (
            "Public speaking is no longer my goal. "
            "I want to become a better leader."
        ),
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert len(changed) == 2
    assert changed[0].id == "goal-1"
    assert changed[0].status is MemoryStatus.DELETED
    assert changed[1].content == (
        "The user wants to become a better leader."
    )
    assert active == (changed[1],)


def test_operation_below_confidence_threshold_is_skipped() -> None:
    service = make_service()
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                CreateMemoryOperation(
                    operation="create",
                    content="The user wants to learn Japanese.",
                    memory_type=MemoryType.GOAL,
                    confidence=0.84,
                    explanation=(
                        "The proposed operation is below policy threshold."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
        confidence_threshold=0.85,
    )

    changed = run_capture(
        manager,
        "I might want to learn Japanese.",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert changed == ()
    assert active == ()


def test_noop_does_not_change_storage() -> None:
    existing = make_memory(
        memory_id="goal-1",
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(existing)
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                NoopMemoryOperation(
                    operation="noop",
                    confidence=0.98,
                    explanation=(
                        "The message contains no durable new information."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )

    changed = run_capture(
        manager,
        "Can you explain how language learning works?",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert changed == ()
    assert active == (existing,)


def test_capture_passes_original_message_to_planner() -> None:
    service = make_service()
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                NoopMemoryOperation(
                    operation="noop",
                    confidence=0.95,
                    explanation=(
                        "The message contains no durable information."
                    ),
                ),
            )
        )
    )
    manager = MemoryManager(
        planner=planner,
        memory_service=service,
        mutation_validator=ApprovingMutationValidator(),
    )
    message = "It rained this afternoon."

    run_capture(manager, message)

    assert planner.user_message == message


@pytest.mark.parametrize("operation_name", ["update", "delete"])
def test_rejected_destructive_operation_is_not_executed(
    operation_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    existing = make_memory(
        memory_id="goal-1",
        content="The user wants to learn piano.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(existing)

    if operation_name == "update":
        operation = UpdateMemoryOperation(
            operation="update",
            memory_index=0,
            content="The user wants to learn guitar.",
            confidence=0.99,
            explanation="Unsafe unrelated replacement.",
        )
    else:
        operation = DeleteMemoryOperation(
            operation="delete",
            memory_index=0,
            confidence=0.99,
            explanation="Unsafe unrelated deletion.",
        )

    validator = RejectingMutationValidator()
    manager = MemoryManager(
        planner=StaticMemoryPlanner(
            MemoryPlan(operations=(operation,))
        ),
        memory_service=service,
        mutation_validator=validator,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="app.memory.manager",
    ):
        changed = run_capture(
            manager,
            "Learning guitar is no longer my goal.",
        )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert changed == ()
    assert active == (existing,)
    assert len(validator.calls) == 1
    assert "Rejected" in caplog.text
    assert operation_name in caplog.text


def test_validator_failure_rejects_only_that_mutation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    existing = make_memory(
        memory_id="goal-1",
        content="The user wants to learn piano.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(existing)
    validator = ScriptedMutationValidator(
        [RuntimeError("validator unavailable")]
    )
    manager = MemoryManager(
        planner=StaticMemoryPlanner(
            MemoryPlan(
                operations=(
                    DeleteMemoryOperation(
                        operation="delete",
                        memory_index=0,
                        confidence=0.99,
                        explanation="Proposed deletion.",
                    ),
                )
            )
        ),
        memory_service=service,
        mutation_validator=validator,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="app.memory.manager",
    ):
        changed = run_capture(
            manager,
            "Piano is no longer my goal.",
        )

    assert changed == ()
    assert asyncio.run(
        service.list_active(user_id=USER_ID)
    ) == (existing,)
    assert "validation failed" in caplog.text
    assert "delete" in caplog.text


def test_multiple_mutations_are_validated_independently() -> None:
    first_goal = make_memory(
        memory_id="goal-1",
        content="The user wants alpha.",
        memory_type=MemoryType.GOAL,
    )
    second_goal = make_memory(
        memory_id="goal-2",
        content="The user wants beta.",
        memory_type=MemoryType.GOAL,
    )
    service = make_service(first_goal, second_goal)
    validator = ScriptedMutationValidator([False, True])
    manager = MemoryManager(
        planner=StaticMemoryPlanner(
            MemoryPlan(
                operations=(
                    UpdateMemoryOperation(
                        operation="update",
                        memory_index=0,
                        content="The user wants gamma.",
                        confidence=0.99,
                        explanation="Proposed replacement.",
                    ),
                    DeleteMemoryOperation(
                        operation="delete",
                        memory_index=1,
                        confidence=0.99,
                        explanation="Proposed deletion.",
                    ),
                )
            )
        ),
        memory_service=service,
        mutation_validator=validator,
    )

    changed = run_capture(
        manager,
        "I want gamma instead of alpha, and beta is no longer my goal.",
    )
    active = asyncio.run(
        service.list_active(user_id=USER_ID)
    )

    assert len(validator.calls) == 2
    assert [call[1].operation for call in validator.calls] == [
        "update",
        "delete",
    ]
    assert len(changed) == 1
    assert changed[0].id == "goal-2"
    assert changed[0].status is MemoryStatus.DELETED
    assert active == (first_goal,)


@pytest.mark.parametrize(
    ("confidence_threshold", "retrieval_limit", "message"),
    [
        (-0.01, 10, "confidence_threshold"),
        (1.01, 10, "confidence_threshold"),
        (0.85, 0, "retrieval_limit"),
        (0.85, -1, "retrieval_limit"),
    ],
)
def test_rejects_invalid_configuration(
    confidence_threshold: float,
    retrieval_limit: int,
    message: str,
) -> None:
    planner = StaticMemoryPlanner(
        MemoryPlan(
            operations=(
                NoopMemoryOperation(
                    operation="noop",
                    confidence=0.95,
                    explanation="No memory change.",
                ),
            )
        )
    )

    with pytest.raises(ValueError, match=message):
        MemoryManager(
            planner=planner,
            memory_service=make_service(),
            mutation_validator=ApprovingMutationValidator(),
            confidence_threshold=confidence_threshold,
            retrieval_limit=retrieval_limit,
        )
