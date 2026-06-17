"""Central application settings loaded from environment variables.

Every service receives configuration through this module so model names,
gateway URLs, Chroma paths, guardrails, Langfuse, and LiteLLM cache settings
stay consistent across local runs and Docker.
"""

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

    # LLM Gateway Config
    llm_gateway_url: str | None = None
    llm_gateway_api_key: str | None = None
    litellm_master_key: str | None = None

    # Langfuse Config
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # LiteLLM Gateway cache config. The app does not use this directly, but the
    # same .env is shared with the LiteLLM container.
    redis_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def gateway_auth_source(self) -> str:
        if self.llm_gateway_api_key:
            return "LLM_GATEWAY_API_KEY"
        if self.litellm_master_key:
            return "LITELLM_MASTER_KEY"
        if self.openai_api_key:
            return "OPENAI_API_KEY_FALLBACK"
        return "NONE"


@lru_cache
def get_settings() -> Settings:
    return Settings()
