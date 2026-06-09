from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    session_id: str | None = Field(default=None, description="Optional session ID for future memory support")


class Source(BaseModel):
    source: str
    chunk_id: str | None = None
    score: float | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source] = []
    metadata: dict[str, Any] = {}
