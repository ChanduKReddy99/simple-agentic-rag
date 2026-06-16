"""Document chunking helpers for retrieval.

This module splits loaded documents into overlapping chunks and assigns stable
chunk IDs so retrieved passages can be traced back in API responses.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{idx}"
    return chunks
