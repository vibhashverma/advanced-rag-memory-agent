"""Pydantic schemas for semantic memory."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: str | None = Field(default=None, description="Memory id (usable for update/delete)")
    fact: str
    category: str = "general"


class UserMemoriesResponse(BaseModel):
    user_id: str
    memories: list[MemoryItem] = Field(default_factory=list)
