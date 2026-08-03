from functools import lru_cache
from pathlib import Path

from memory_engine.config import get_settings
from memory_engine.embeddings import EmbeddingClient, OllamaEmbeddingClient
from memory_engine.llm import LLMClient, OllamaLLMClient, OpenAIResponsesLLMClient
from memory_engine.memory.context_retriever import (
    ContextMemoryRetriever,
    FinalContextMemoryRetriever,
)
from memory_engine.memory.extractor import (
    LLMMemoryFactExtractor,
    MemoryFactExtractor,
)
from memory_engine.memory.reconciler import LLMMemoryReconciler, MemoryReconciler
from memory_engine.memory.reranker import LLMMemoryReranker, MemoryReranker
from memory_engine.memory.retriever import MemoryRetriever, SemanticCandidateRetriever
from memory_engine.memory.semantic_manager import SemanticMemoryManager
from memory_engine.memory.service import MemoryService
from memory_engine.memory.sqlite_service import SQLiteMemoryService
from memory_engine.memory.validator import (
    LLMMemoryMutationValidator,
    MemoryMutationValidator,
)
from memory_engine.prompting import PromptRepository

REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def get_prompt_repository() -> PromptRepository:
    return PromptRepository(REPO_ROOT / "memory_engine" / "prompts")


def _require_api_key() -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return settings.openai_api_key


@lru_cache
def get_semantic_llm() -> LLMClient:
    """
    Generic configurable model client used only by evaluation scripts to
    compare candidates (e.g. a local model vs. a hosted one) before a
    boundary's fixed assignment is decided. Production code always uses
    the per-boundary getters below instead — see "Fixed model
    assignment, not runtime semantic escalation" in docs/architecture.md.
    """
    settings = get_settings()
    if settings.semantic_provider.casefold() == "openai":
        return OpenAIResponsesLLMClient(
            api_key=_require_api_key(),
            model=settings.semantic_model,
            base_url=settings.openai_base_url,
            reasoning_effort=settings.semantic_reasoning_effort,
            timeout_seconds=settings.semantic_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_backoff_seconds=settings.llm_retry_backoff_seconds,
        )
    return OllamaLLMClient(
        model=settings.semantic_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        json_mode=True,
        timeout_seconds=settings.semantic_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )


@lru_cache
def get_memory_llm() -> LLMClient:
    """Evaluation-only default: local Ollama unless explicitly switched to hosted."""
    settings = get_settings()
    if settings.memory_use_semantic_model:
        return get_semantic_llm()
    return OllamaLLMClient(
        model=settings.memory_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        json_mode=True,
        timeout_seconds=settings.semantic_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )


@lru_cache
def get_extraction_llm() -> LLMClient:
    settings = get_settings()
    return OpenAIResponsesLLMClient(
        api_key=_require_api_key(),
        model=settings.extraction_model,
        base_url=settings.openai_base_url,
        reasoning_effort=settings.extraction_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )


@lru_cache
def get_reconciliation_llm() -> LLMClient:
    settings = get_settings()
    return OpenAIResponsesLLMClient(
        api_key=_require_api_key(),
        model=settings.reconciliation_model,
        base_url=settings.openai_base_url,
        reasoning_effort=settings.reconciliation_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )


@lru_cache
def get_validation_llm() -> LLMClient:
    settings = get_settings()
    return OpenAIResponsesLLMClient(
        api_key=_require_api_key(),
        model=settings.validation_model,
        base_url=settings.openai_base_url,
        reasoning_effort=settings.validation_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )


@lru_cache
def get_reranking_llm() -> LLMClient:
    settings = get_settings()
    return OpenAIResponsesLLMClient(
        api_key=_require_api_key(),
        model=settings.reranking_model,
        base_url=settings.openai_base_url,
        reasoning_effort=settings.reranking_reasoning_effort,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
    )


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    settings = get_settings()
    return OllamaEmbeddingClient(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
    )


@lru_cache
def get_memory_service() -> MemoryService:
    settings = get_settings()
    db_path = REPO_ROOT / settings.memory_db_path
    return SQLiteMemoryService(db_path)


@lru_cache
def get_memory_extractor() -> MemoryFactExtractor:
    return LLMMemoryFactExtractor(
        llm_client=get_extraction_llm(),
        prompt_repository=get_prompt_repository(),
    )


@lru_cache
def get_memory_reconciler() -> MemoryReconciler:
    return LLMMemoryReconciler(
        llm_client=get_reconciliation_llm(),
        prompt_repository=get_prompt_repository(),
    )


@lru_cache
def get_memory_mutation_validator() -> MemoryMutationValidator:
    return LLMMemoryMutationValidator(
        llm_client=get_validation_llm(),
        prompt_repository=get_prompt_repository(),
    )


@lru_cache
def get_memory_reranker() -> MemoryReranker:
    return LLMMemoryReranker(
        llm_client=get_reranking_llm(),
        prompt_repository=get_prompt_repository(),
    )


@lru_cache
def get_memory_retriever() -> MemoryRetriever:
    settings = get_settings()
    return SemanticCandidateRetriever(
        embedding_client=get_embedding_client(),
        technical_floor=settings.memory_candidate_technical_floor,
    )


@lru_cache
def get_context_memory_retriever() -> FinalContextMemoryRetriever:
    settings = get_settings()
    return ContextMemoryRetriever(
        candidate_retriever=get_memory_retriever(),
        reranker=get_memory_reranker(),
        candidate_limit=settings.memory_retrieval_limit,
    )


@lru_cache
def get_memory_manager() -> SemanticMemoryManager:
    settings = get_settings()
    return SemanticMemoryManager(
        extractor=get_memory_extractor(),
        reconciler=get_memory_reconciler(),
        memory_service=get_memory_service(),
        memory_retriever=get_memory_retriever(),
        mutation_validator=get_memory_mutation_validator(),
        confidence_threshold=settings.memory_confidence_threshold,
        retrieval_limit=settings.memory_retrieval_limit,
    )
