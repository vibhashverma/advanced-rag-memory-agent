"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.generation import GenerationError
from app.api import chat, documents, memories
from app.core.config import ConfigurationError, get_settings
from app.core.logging import configure_logging, get_logger, kv
from app.core.security import redact
from app.memory.mem0_manager import MemoryError_
from app.services.retrieval_service import RetrievalError

log = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Advanced RAG Agent with Context Compression & Semantic Memory",
        version="1.0.0",
        description=(
            "Hybrid (dense + BM25) retrieval with RRF, parent-document retrieval, Cohere Rerank "
            "context compression, Mem0 long-term memory and dynamic persona adaptation."
        ),
    )
    app.include_router(chat.router)
    app.include_router(memories.router)
    app.include_router(documents.router)

    @app.get("/health", tags=["system"], summary="Liveness check")
    async def health() -> dict:
        return {"status": "ok"}

    def _err(status: int, exc: Exception, public: str | None = None) -> JSONResponse:
        log.error(kv(event="request_error", status=status, error=type(exc).__name__))
        return JSONResponse(status_code=status, content={"detail": redact(public or str(exc))})

    @app.exception_handler(ConfigurationError)
    async def _config(_: Request, exc: ConfigurationError) -> JSONResponse:
        return _err(503, exc)

    @app.exception_handler(RetrievalError)
    async def _retrieval(_: Request, exc: RetrievalError) -> JSONResponse:
        return _err(502, exc)

    @app.exception_handler(GenerationError)
    async def _generation(_: Request, exc: GenerationError) -> JSONResponse:
        return _err(502, exc)

    @app.exception_handler(MemoryError_)
    async def _memory(_: Request, exc: MemoryError_) -> JSONResponse:
        return _err(502, exc)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        return _err(500, exc, "Internal server error")

    return app


app = create_app()
