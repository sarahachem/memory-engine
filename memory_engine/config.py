from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    extraction_model: str = "gpt-5.4-mini"
    extraction_reasoning_effort: str = "low"
    reconciliation_model: str = "gpt-5.6-terra"
    reconciliation_reasoning_effort: str = "low"
    validation_model: str = "gpt-5.6-terra"
    validation_reasoning_effort: str = "low"
    reranking_model: str = "gpt-5.4-mini"
    reranking_reasoning_effort: str = "low"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma"

    memory_confidence_threshold: float = 0.85
    memory_retrieval_limit: int = 10
    memory_candidate_technical_floor: float = 0.0
    memory_db_path: str = "data/memory.sqlite3"

    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.25

    # Used only by evaluation scripts to compare a cheap local model
    # against the fixed hosted assignments above — never by production
    # dependency wiring, which always uses the per-boundary getters.
    semantic_provider: str = "openai"
    semantic_model: str = "gpt-5.6-sol"
    semantic_reasoning_effort: str = "low"
    semantic_timeout_seconds: float = 120.0
    memory_use_semantic_model: bool = False
    memory_model: str = "mistral-small3.2:24b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
