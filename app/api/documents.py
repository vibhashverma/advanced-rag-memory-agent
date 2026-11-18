"""POST /api/documents - upload a PDF/TXT and ingest it immediately."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_indexer
from app.core.config import get_settings
from app.ingestion.indexing import Indexer
from app.ingestion.loader import SUPPORTED_EXTENSIONS
from app.schemas.chat import ErrorResponse
from app.schemas.documents import UploadResponse

router = APIRouter(prefix="/api", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@router.post(
    "/documents",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Upload a PDF or TXT file and add it to the knowledge base",
)
async def upload_document(
    file: UploadFile = File(..., description="A .pdf or .txt file"),
    provider: Callable[[], Indexer] = Depends(get_indexer),
) -> UploadResponse:
    name = Path(file.filename or "").name  # strip any client-supplied directories
    ext = Path(name).suffix.lower()
    if not name or ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'!r}; upload one of {sorted(SUPPORTED_EXTENSIONS)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    indexer = await run_in_threadpool(provider)
    docs_dir = get_settings().documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / name
    target.write_bytes(data)
    result = await run_in_threadpool(indexer.ingest_file, target)
    if result.status == "failed":
        target.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not ingest {name}: {result.error}")
    return UploadResponse(
        filename=name, document_id=result.document_id, status=result.status,
        parents=result.parents, children=result.children,
    )
