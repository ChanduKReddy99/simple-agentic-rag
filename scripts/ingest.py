import argparse

from app.core.config import get_settings
from app.core.logging import configure_logging, log_detail, log_step_finish, log_step_start
from app.rag.loader import load_markdown_documents
from app.rag.chunker import chunk_documents
from app.rag.embeddings import build_embeddings
from app.rag.vectorstore import build_vectorstore, ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local markdown documents into Chroma.")
    parser.parse_args()
    run_ingestion(configure_run_logging=True)


def run_ingestion(configure_run_logging: bool = False) -> None:
    settings = get_settings()
    if configure_run_logging:
        configure_logging(settings.log_level, run_name="document_ingestion")

    log_step_start("load_documents", "source=data")
    raw_docs = load_markdown_documents("data")
    log_step_finish("load_documents", f"documents={len(raw_docs)}")

    log_step_start("chunk_documents", f"chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    chunks = chunk_documents(
        docs=raw_docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    log_step_finish("chunk_documents", f"chunks={len(chunks)}")

    log_step_start("build_embeddings", f"model={settings.openai_embedding_model}")
    embeddings = build_embeddings(settings)
    log_step_finish("build_embeddings")

    log_step_start("build_vectorstore", f"collection={settings.chroma_collection}")
    vectorstore = build_vectorstore(settings, embeddings)
    log_step_finish("build_vectorstore", f"persist_dir={settings.chroma_persist_dir}")

    log_step_start("ingest_documents", f"chunks={len(chunks)}")
    ingest_documents(vectorstore, chunks)
    log_step_finish("ingest_documents", f"collection={settings.chroma_collection}")

    log_detail("ingested_chunks", len(chunks))
    print(f"Ingested {len(chunks)} chunks into Chroma collection '{settings.chroma_collection}'")


if __name__ == "__main__":
    main()
