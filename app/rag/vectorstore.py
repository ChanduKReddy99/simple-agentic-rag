from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from app.core.config import Settings


def build_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def ingest_documents(vectorstore: Chroma, documents: list[Document]) -> None:
    if not documents:
        return
    ids = [f"{doc.metadata.get('source', 'doc')}::{doc.metadata.get('chunk_id', idx)}" for idx, doc in enumerate(documents)]
    vectorstore.add_documents(documents=documents, ids=ids)
