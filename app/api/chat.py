"""POST /api/chat"""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse, ErrorResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Ask a question (hybrid RAG + semantic memory + persona)",
)
async def chat(
    req: ChatRequest, provider: Callable[[], ChatService] = Depends(get_chat_service)
) -> ChatResponse:
    # The pipeline uses blocking SDKs, so run it off the event loop.
    service = await run_in_threadpool(provider)  # may build clients on first call
    return await run_in_threadpool(service.handle, req)
