from langchain_core.documents import Document
from app.rag.chunker import chunk_documents


def test_chunk_documents_adds_chunk_ids():
    docs = [Document(page_content="hello world " * 200, metadata={"source": "test.md"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
