from __future__ import annotations

from memory_engine.config import get_settings
from memory_engine.llm import LLMClient, OpenAIResponsesLLMClient


def build_openai_evaluation_client(model: str) -> LLMClient:
    """Build an uncached, explicitly attributed OpenAI evaluator client."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for an explicit evaluation model."
        )
    return OpenAIResponsesLLMClient(
        api_key=settings.openai_api_key,
        model=model,
        base_url=settings.openai_base_url,
        reasoning_effort=settings.semantic_reasoning_effort,
        timeout_seconds=settings.semantic_timeout_seconds,
    )
