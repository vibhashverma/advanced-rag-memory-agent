"""Retrieval pipeline: hybrid (dense+sparse+RRF) -> parents (top-20) -> Cohere rerank -> top-5."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.compression.reranker import RerankError, ThresholdCohereRerank
from app.core.logging import get_logger, kv
from app.core.security import redact
from app.retrieval.parent_retriever import HybridParentRetriever, RetrievalStats
from app.services.telemetry_service import RequestTimer

log = get_logger(__name__)


class RetrievalError(RuntimeError):
    """Retrieval backend (embeddings / vector store) failed; message is safe to return."""


@dataclass
class RetrievalOutcome:
    contexts: list[Document]  # compressed contexts, at most `top_n`
    stats: RetrievalStats
    candidates: int  # documents handed to the reranker
    rerank_degraded: bool = False  # True if Cohere failed and RRF order was used instead


class RetrievalService:
    def __init__(
        self,
        retriever: HybridParentRetriever,
        compressor: ThresholdCohereRerank,
        secrets: list[str] | None = None,
    ) -> None:
        self.retriever = retriever
        self.compressor = compressor
        self._secrets = secrets or []

    def retrieve(self, query: str, timer: RequestTimer) -> RetrievalOutcome:
        try:
            with timer.stage("retrieval"):
                candidates, stats = self.retriever.retrieve_with_stats(query)
        except Exception as exc:
            safe = redact(f"{type(exc).__name__}: {exc}", self._secrets)
            log.error(kv(event="retrieval_failed", error=safe))
            raise RetrievalError(f"Document retrieval failed ({type(exc).__name__}).") from None

        degraded = False
        contexts: list[Document] = []
        with timer.stage("rerank"):
            if candidates:
                try:
                    contexts = list(self.compressor.compress_documents(candidates, query))
                except RerankError as exc:
                    # Graceful degradation: keep RRF order, flag it (never silently).
                    degraded = True
                    contexts = candidates[: self.compressor.top_n]
                    log.warning(kv(event="rerank_degraded", error=str(exc)))
        log.info(
            kv(
                event="retrieval_done",
                dense=stats.dense_count,
                sparse=stats.sparse_count,
                candidates=len(candidates),
                kept=len(contexts),
                degraded=degraded,
            )
        )
        return RetrievalOutcome(contexts=contexts, stats=stats, candidates=len(candidates), rerank_degraded=degraded)
