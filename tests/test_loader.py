"""Tests for loading markdown knowledge-base documents."""

from app.rag.loader import load_markdown_documents


def test_load_markdown_documents():
    docs = load_markdown_documents("data")
    assert len(docs) >= 1
    assert all(doc.page_content for doc in docs)
    assert all("source" in doc.metadata for doc in docs)
