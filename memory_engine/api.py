from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from memory_engine.dependencies import (
    get_context_memory_retriever,
    get_memory_manager,
    get_memory_service,
)
from memory_engine.memory.context_retriever import FinalContextMemoryRetriever
from memory_engine.memory.semantic_manager import SemanticMemoryManager
from memory_engine.memory.service import Memory, MemoryService
from memory_engine.models import Intent, MemoryType

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    intent: Intent = Intent.GENERAL_CONVERSATION


class MemoryResponse(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    confidence: float
    source_quote: str | None = None
    created_at: datetime | None = None


class RelevantMemoryResponse(BaseModel):
    memory: MemoryResponse
    score: float


class ChatResponse(BaseModel):
    """
    Demonstrates both memory paths in one call: capture (write) runs the
    full extract -> reconcile -> validate -> store pipeline against the
    message; retrieve (read) then reranks the user's active memories
    against that same message, exactly as a real response-composition
    step would. See docs/architecture.md.
    """

    changed_memories: list[MemoryResponse]
    relevant_memories: list[RelevantMemoryResponse]


class MemoryUpdateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2_000)


def _to_memory_response(memory: Memory) -> MemoryResponse:
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        memory_type=memory.memory_type,
        confidence=memory.confidence,
        source_quote=memory.evidence,
        created_at=memory.created_at,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    memory_manager: SemanticMemoryManager = Depends(get_memory_manager),
    memory_service: MemoryService = Depends(get_memory_service),
    context_retriever: FinalContextMemoryRetriever = Depends(
        get_context_memory_retriever
    ),
) -> ChatResponse:
    changed = await memory_manager.capture(
        user_id=request.user_id,
        user_message=request.message,
    )
    active_memories = await memory_service.list_active(request.user_id)
    relevant = await context_retriever.retrieve(
        message=request.message,
        intent=request.intent,
        candidate_memories=active_memories,
        limit=5,
    )
    return ChatResponse(
        changed_memories=[_to_memory_response(memory) for memory in changed],
        relevant_memories=[
            RelevantMemoryResponse(
                memory=_to_memory_response(result.memory),
                score=result.score,
            )
            for result in relevant
        ],
    )


@router.get("/memory/{user_id}", response_model=list[MemoryResponse])
async def list_memories(
    user_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> list[MemoryResponse]:
    memories = await memory_service.list_active(user_id)
    return [_to_memory_response(memory) for memory in memories]


@router.patch("/memory/{user_id}/{memory_id}", response_model=MemoryResponse)
async def correct_memory(
    user_id: str,
    memory_id: str,
    request: MemoryUpdateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    existing = next(
        (
            memory
            for memory in await memory_service.list_active(user_id)
            if memory.id == memory_id
        ),
        None,
    )
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active memory not found: {memory_id}",
        )
    try:
        memory = await memory_service.update(
            user_id=user_id,
            memory_id=memory_id,
            content=request.content,
            # A user-authored correction is ground truth, not a model
            # guess — it replaces whatever confidence the extractor
            # originally assigned.
            confidence=1.0,
            source_episode_id=existing.source_episode_id,
            evidence=existing.evidence,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return _to_memory_response(memory)


@router.delete(
    "/memory/{user_id}/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_memory(
    user_id: str,
    memory_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> None:
    """True erasure, not a tombstone — see docs/architecture.md."""
    try:
        await memory_service.purge(user_id=user_id, memory_id=memory_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.delete("/memory/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_memories(
    user_id: str,
    memory_service: MemoryService = Depends(get_memory_service),
) -> None:
    """Erases every stored memory for one user — a full right-to-erasure request."""
    await memory_service.purge_user(user_id=user_id)
