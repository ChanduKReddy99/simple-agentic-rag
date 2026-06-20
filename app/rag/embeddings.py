"""Embedding model factory.

This module centralizes creation of OpenAI embeddings so ingestion and retrieval
use the same configured embedding model.
"""

from langchain_openai import OpenAIEmbeddings
from app.core.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    llm_api_key = (
        settings.llm_gateway_api_key
        or settings.litellm_master_key
        or settings.openai_api_key
    )
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=llm_api_key,
        base_url=settings.llm_gateway_url,
    )
