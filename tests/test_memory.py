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
from memory_engine.memory.service import InMemoryMemoryService, Memory, MemoryStatus
from memory_engine.memory.validator import (
    MemoryMutationValidation,
    MemoryMutationValidator,
)
from memory_engine.models import MemoryType


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


class StubPlanner(MemoryPlanner):
    def __init__(self, plan: MemoryPlan) -> None:
        self.result = plan
        self.received_memories: tuple[Memory, ...] = ()

    async def plan(
        self,
        user_message: str,
        active_memories: tuple[Memory, ...],
    ) -> MemoryPlan:
        self.received_memories = active_memories
        return self.result


def operation(operation_type: str, **values: object):
    common = {"confidence": 0.95, "explanation": "test"}
    common.update(values)
    types = {
        "create": CreateMemoryOperation,
        "update": UpdateMemoryOperation,
        "delete": DeleteMemoryOperation,
        "noop": NoopMemoryOperation,
    }
    return types[operation_type](operation=operation_type, **common)


@pytest.mark.asyncio
async def test_manager_applies_delete_and_create() -> None:
    old = Memory(
        id="old-goal",
        content="The user wants to speak in public.",
        memory_type=MemoryType.GOAL,
        confidence=0.95,
    )
    service = InMemoryMemoryService({"user": (old,)})
    planner = StubPlanner(
        MemoryPlan(
            operations=(
                operation("delete", memory_index=0),
                operation(
                    "create",
                    content="The user wants to become a better leader.",
                    memory_type=MemoryType.GOAL,
                ),
            )
        )
    )

    changed = await MemoryManager(
        planner,
        service,
        ApprovingMutationValidator(),
    ).capture(
        "user",
        "Public speaking is no longer my goal; I want to become a better leader.",
    )

    assert [memory.status for memory in changed] == [
        MemoryStatus.DELETED,
        MemoryStatus.ACTIVE,
    ]
    active = await service.list_active("user")
    assert [memory.content for memory in active] == [
        "The user wants to become a better leader."
    ]


@pytest.mark.asyncio
async def test_manager_updates_memory_without_changing_type() -> None:
    existing = Memory(
        id="location",
        content="The user lives in Munich.",
        memory_type=MemoryType.PERSONAL_FACT,
        confidence=0.9,
    )
    service = InMemoryMemoryService({"user": (existing,)})
    planner = StubPlanner(
        MemoryPlan(
            operations=(
                operation(
                    "update",
                    memory_index=0,
                    content="The user lives in Berlin.",
                ),
            )
        )
    )

    changed = await MemoryManager(
        planner,
        service,
        ApprovingMutationValidator(),
    ).capture(
        "user", "I now live in Berlin, not Munich."
    )

    assert changed[0].memory_type == MemoryType.PERSONAL_FACT
    assert changed[0].content == "The user lives in Berlin."


@pytest.mark.asyncio
async def test_duplicate_and_low_confidence_operations_are_skipped() -> None:
    existing = Memory(
        id="goal",
        content="The user wants to learn Japanese.",
        memory_type=MemoryType.GOAL,
        confidence=0.95,
    )
    service = InMemoryMemoryService({"user": (existing,)})
    planner = StubPlanner(
        MemoryPlan(
            operations=(
                operation(
                    "create",
                    content="The user wants to learn Japanese.",
                    memory_type=MemoryType.GOAL,
                ),
                CreateMemoryOperation(
                    operation="create",
                    content="A low confidence fact",
                    memory_type=MemoryType.PERSONAL_FACT,
                    confidence=0.2,
                    explanation="uncertain",
                ),
            )
        )
    )

    changed = await MemoryManager(
        planner,
        service,
        ApprovingMutationValidator(),
    ).capture(
        "user", "I want to learn Japanese."
    )

    assert changed == ()
    assert await service.list_active("user") == (existing,)


@pytest.mark.asyncio
async def test_search_excludes_deleted_and_is_deterministic() -> None:
    memories = (
        Memory(
            id="first",
            content="The user wants to learn Japanese.",
            memory_type=MemoryType.GOAL,
            confidence=0.9,
        ),
        Memory(
            id="deleted",
            content="The user wants to learn Japanese grammar.",
            memory_type=MemoryType.GOAL,
            confidence=0.9,
            status=MemoryStatus.DELETED,
        ),
        Memory(
            id="unrelated",
            content="The user lives in Berlin.",
            memory_type=MemoryType.PERSONAL_FACT,
            confidence=0.9,
        ),
    )
    service = InMemoryMemoryService({"user": memories})

    result = await service.search("user", "I want to learn Japanese", limit=1)

    assert [memory.id for memory in result] == ["first"]
