"""Chat orchestration: memory -> persona -> retrieval -> rerank -> generation -> memory update."""
from __future__ import annotations

from app.agent.rag_agent import RagAgent
from app.core.logging import get_logger, kv
from app.memory.mem0_manager import Mem0Manager
from app.memory.persona import UserPersona, build_persona
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem
from app.services.retrieval_service import RetrievalService
from app.services.telemetry_service import RequestTimer

log = get_logger(__name__)


class ChatService:
    def __init__(self, memory: Mem0Manager, retrieval: RetrievalService, agent: RagAgent) -> None:
        self.memory = memory
        self.retrieval = retrieval
        self.agent = agent

    def handle(self, req: ChatRequest) -> ChatResponse:
        timer = RequestTimer()

        # 1. Long-term memory -> persona. Keyed by user_id only, so it spans sessions.
        persona = UserPersona(user_id=req.user_id)
        try:
            with timer.stage("memory_retrieval"):
                persona = build_persona(req.user_id, self.memory.get_all_memories(req.user_id))
        except Exception as exc:  # memory outage must not take chat down
            log.warning(kv(event="memory_unavailable", user_id=req.user_id, error=type(exc).__name__))

        # 2-4. Hybrid retrieval -> parents -> rerank/compress.
        outcome = self.retrieval.retrieve(req.query, timer)

        # 5. Generate.
        with timer.stage("generation"):
            answer = self.agent.answer(req.query, persona, outcome.contexts)

        # 6. Extract + persist durable facts from this user message.
        saved = 0
        try:
            with timer.stage("memory_update"):
                saved = len(self.memory.add_memory(req.user_id, req.query, session_id=req.session_id))
        except Exception as exc:
            log.warning(kv(event="memory_update_failed", user_id=req.user_id, error=type(exc).__name__))

        telemetry = timer.snapshot(
            dense_hits=outcome.stats.dense_count,
            sparse_hits=outcome.stats.sparse_count,
            retrieved_candidates=outcome.candidates,
            compressed_contexts=len(outcome.contexts),
            rerank_degraded=outcome.rerank_degraded,
            memories_used=persona.fact_count,
            memories_saved=saved,
        )
        log.info(kv(event="chat_done", user_id=req.user_id, session_id=req.session_id, total_ms=telemetry.total_latency_ms))
        return ChatResponse(
            answer=answer,
            user_id=req.user_id,
            session_id=req.session_id,
            sources=[_source(d) for d in outcome.contexts],
            telemetry=telemetry,
        )


def _source(doc) -> SourceItem:
    m = doc.metadata
    return SourceItem(
        source=m.get("source"),
        page=m.get("page"),
        parent_id=m.get("parent_id"),
        relevance_score=m.get("relevance_score"),
        snippet=doc.page_content.strip()[:200],
    )
