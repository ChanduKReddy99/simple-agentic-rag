"""LangGraph workflow that powers the Agentic RAG behavior.

The graph rewrites user questions, retrieves relevant Chroma documents, decides
whether the context is strong enough, and then either answers from context or
uses a safe fallback prompt.
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.agent.state import AgentState
from app.agent.prompts import QUERY_REWRITE_PROMPT, ANSWER_PROMPT, FALLBACK_PROMPT
from app.core.config import Settings
from app.rag.retriever import RetrieverService


class AgenticRAGGraph:
    def __init__(self, retriever: RetrieverService, settings: Settings):
        self.retriever = retriever
        self.settings = settings
        llm_api_key = (
            settings.llm_gateway_api_key
            or settings.litellm_master_key
            or settings.openai_api_key
        )
        self.llm = ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=llm_api_key,
            base_url=settings.llm_gateway_url,
            temperature=0.1,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer_from_context", self.answer_from_context)
        workflow.add_node("fallback_answer", self.fallback_answer)

        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_retrieval,
            {
                "answer_from_context": "answer_from_context",
                "fallback_answer": "fallback_answer",
            },
        )
        workflow.add_edge("answer_from_context", END)
        workflow.add_edge("fallback_answer", END)
        return workflow.compile()

    def rewrite_query(self, state: AgentState) -> AgentState:
        question = state["question"]
        prompt = QUERY_REWRITE_PROMPT.format(question=question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"rewritten_question": response.content.strip() or question}

    def retrieve(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_question") or state["question"]
        docs = self.retriever.search(query)
        return {"retrieved_docs": docs}

    def route_after_retrieval(self, state: AgentState) -> str:
        docs = state.get("retrieved_docs", [])
        if not docs:
            return "fallback_answer"
        best_score = max(doc.score for doc in docs)
        if best_score >= self.settings.min_retrieval_score:
            return "answer_from_context"
        return "fallback_answer"

    def answer_from_context(self, state: AgentState) -> AgentState:
        docs = state.get("retrieved_docs", [])
        context = "\n\n".join(
            f"Source: {d.document.metadata.get('source')}\n{d.document.page_content}"
            for d in docs
        )
        prompt = ANSWER_PROMPT.format(context=context, question=state["question"])
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "answer": response.content.strip(),
            "route": "rag_context",
            "sources": self._format_sources(docs),
            "contexts": self._format_contexts(docs),
        }

    def fallback_answer(self, state: AgentState) -> AgentState:
        prompt = FALLBACK_PROMPT.format(question=state["question"])
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "answer": response.content.strip(),
            "route": "fallback_no_strong_context",
            "sources": self._format_sources(state.get("retrieved_docs", [])),
            "contexts": self._format_contexts(state.get("retrieved_docs", [])),
        }

    @staticmethod
    def _format_sources(docs):
        return [
            {
                "source": d.document.metadata.get("source", "unknown"),
                "chunk_id": d.document.metadata.get("chunk_id"),
                "score": d.score,
                "preview": d.document.page_content[:250].replace("\n", " "),
            }
            for d in docs
        ]

    @staticmethod
    def _format_contexts(docs):
        return [d.document.page_content for d in docs]

    def invoke(self, question: str, config: dict | None = None) -> AgentState:
        return self.graph.invoke({"question": question}, config=config)
