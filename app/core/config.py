from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Agentic RAG System"
    app_env: str = "local"
    log_level: str = "INFO"

    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_persist_dir: str = "storage/chroma"
    chroma_collection: str = "agentic_rag_docs"

    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    min_retrieval_score: float = 0.35
    guardrails_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
