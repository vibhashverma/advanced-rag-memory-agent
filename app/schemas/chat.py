"""Pydantic schemas for the chat API."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ID_PATTERN = r"^[A-Za-z0-9_.:@\-]{1,128}$"
IdStr = Annotated[str, StringConstraints(strip_whitespace=True, pattern=ID_PATTERN)]


class ChatRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    user_id: IdStr
    session_id: IdStr

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "Explain hybrid search", "user_id": "usr_99", "session_id": "session_001"}
        }
    )


class SourceItem(BaseModel):
    source: str | None = None
    page: int | None = None
    parent_id: str | None = None
    relevance_score: float | None = None
    snippet: str = ""


class Telemetry(BaseModel):
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    # Additional measured stages / debugging counters
    memory_retrieval_latency_ms: float = 0.0
    memory_update_latency_ms: float = 0.0
    dense_hits: int = 0
    sparse_hits: int = 0
    retrieved_candidates: int = 0
    compressed_contexts: int = 0
    rerank_degraded: bool = False
    memories_used: int = 0
    memories_saved: int = 0


class ChatResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str
    sources: list[SourceItem] = Field(default_factory=list)
    telemetry: Telemetry


class ErrorResponse(BaseModel):
    detail: str
