"""Document loading helpers for the local knowledge base.

This file reads markdown files from `data/` and converts them into LangChain
Document objects with source metadata for later citation.
"""

from pathlib import Path
from langchain_core.documents import Document


def load_markdown_documents(data_dir: str = "data") -> list[Document]:
    docs: list[Document] = []
    for path in Path(data_dir).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs
