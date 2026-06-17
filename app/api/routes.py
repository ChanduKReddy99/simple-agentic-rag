"""HTTP routes for health checks and chat requests.

This file validates incoming questions, invokes the Agentic RAG graph, attaches
optional Langfuse callbacks, runs output guardrails, and shapes API responses.
"""

from fastapi import APIRouter, Request, HTTPException
from app.core.config import get_settings
from app.core.guardrails import validate_answer, validate_question
from app.core.logging import log_detail, log_step_finish, log_step_start
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    settings = get_settings()
    if settings.guardrails_enabled:
        log_step_start("input_guardrails", "chat_request")
        input_guardrail = validate_question(payload.question)
        log_detail("input_allowed", input_guardrail.allowed)
        log_detail("input_reason", input_guardrail.reason)
        if not input_guardrail.allowed:
            log_step_finish("input_guardrails", "blocked")
            return ChatResponse(
                answer=input_guardrail.message or "I can't process that request.",
                route="guardrail_input_blocked",
                sources=[],
                metadata={
                    "session_id": payload.session_id,
                    **input_guardrail.metadata,
                },
            )
        log_step_finish("input_guardrails", "passed")

    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")

    config = {}
    if (
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and getattr(request.app.state, "langfuse_client", None) is not None
    ):
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [
            CallbackHandler(public_key=settings.langfuse_public_key)
        ]
        if payload.session_id:
            config["metadata"] = {"langfuse_session_id": payload.session_id}

    log_step_start("agent_invoke")
    result = agent.invoke(payload.question, config=config)
    log_detail("route", result.get("route", "unknown"))
    log_step_finish("agent_invoke")

    if settings.guardrails_enabled:
        log_step_start("output_guardrails", "chat_response")
        output_guardrail = validate_answer(result.get("answer", ""))
        log_detail("output_allowed", output_guardrail.allowed)
        log_detail("output_reason", output_guardrail.reason)
        if not output_guardrail.allowed:
            log_step_finish("output_guardrails", "blocked")
            return ChatResponse(
                answer=output_guardrail.message or "I blocked this response.",
                route="guardrail_output_blocked",
                sources=[],
                metadata={
                    "session_id": payload.session_id,
                    "original_route": result.get("route", "unknown"),
                    **output_guardrail.metadata,
                },
            )
        log_step_finish("output_guardrails", "passed")

    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", "unknown"),
        sources=result.get("sources", []),
        metadata={
            "session_id": payload.session_id,
            "rewritten_question": result.get("rewritten_question"),
            "guardrails_enabled": settings.guardrails_enabled,
        },
    )
