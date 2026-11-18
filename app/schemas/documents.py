"""Pydantic schemas for document upload."""
from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    document_id: str
    status: str  # "ingested" | "skipped" (identical content already indexed)
    parents: int = 0
    children: int = 0
