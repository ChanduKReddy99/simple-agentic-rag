"""Prompt templates used by the Agentic RAG workflow.

The graph imports these templates for query rewriting, answering from retrieved
context, and producing a safe fallback answer when context is weak.
"""

QUERY_REWRITE_PROMPT = """
You are a query rewriting assistant for a production RAG system.
Rewrite the user's question into a precise search query.
Keep project names, tool names, and technical terms.
Return only the rewritten query.

Question: {question}
"""

ANSWER_PROMPT = """
You are a senior AI engineer answering using only the provided context.
Be accurate, practical, and concise.
If the context is insufficient, say what is missing instead of guessing.

Context:
{context}

Question:
{question}

Answer:
"""

FALLBACK_PROMPT = """
You are an AI assistant. The retrieval system did not find enough reliable context.
Answer cautiously and explain that the local knowledge base did not contain strong evidence.

Question:
{question}
"""
