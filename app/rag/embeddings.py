"""Embedding model factory.

This module centralizes creation of OpenAI embeddings so ingestion and retrieval
use the same configured embedding model.
"""

from langchain_openai import OpenAIEmbeddings
from app.core.config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
