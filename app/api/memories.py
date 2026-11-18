"""GET /api/users/{user_id}/memories"""
from __future__ import annotations

from typing import Annotated, Callable

from fastapi import APIRouter, Depends, Path
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_memory_manager
from app.memory.mem0_manager import Mem0Manager
from app.schemas.chat import ID_PATTERN, ErrorResponse
from app.schemas.memory import UserMemoriesResponse

router = APIRouter(prefix="/api", tags=["memory"])


@router.get(
    "/users/{user_id}/memories",
    response_model=UserMemoriesResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="List the stored semantic memories for one user",
)
async def list_memories(
    user_id: Annotated[str, Path(pattern=ID_PATTERN, description="User identifier", examples=["usr_99"])],
    provider: Callable[[], Mem0Manager] = Depends(get_memory_manager),
) -> UserMemoriesResponse:
    manager = await run_in_threadpool(provider)
    memories = await run_in_threadpool(manager.get_all_memories, user_id)
    return UserMemoriesResponse(user_id=user_id, memories=memories)
