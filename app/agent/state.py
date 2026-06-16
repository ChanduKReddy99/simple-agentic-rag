"""Typed state shared between LangGraph workflow nodes.

Each key represents data produced or consumed as a question moves through query
rewriting, retrieval, routing, answering, and response formatting.
"""

from typing import TypedDict, Any
from app.rag.retriever import RetrievedDocument


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    retrieved_docs: list[RetrievedDocument]
    route: str
    answer: str
    sources: list[dict[str, Any]]
    contexts: list[str]
