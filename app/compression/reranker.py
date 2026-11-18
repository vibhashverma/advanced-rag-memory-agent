"""Cohere Rerank context compression: rerank candidates, drop low-score ones, keep top N."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.core.config import Settings
from app.core.logging import get_logger, kv
from app.core.security import redact

log = get_logger(__name__)


class RerankError(RuntimeError):
    """The rerank provider failed. Message is already redacted of secrets."""


class ThresholdCohereRerank(BaseDocumentCompressor):
    """Wraps LangChain's CohereRerank, then enforces a minimum relevance score and a hard top-N cap.

    Output documents are sorted by `relevance_score` (descending) and keep all source metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reranker: Any  # langchain_cohere.CohereRerank (anything exposing .rerank(documents, query, top_n=...))
    top_n: int = 5
    min_score: float = 0.0

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        try:
            results = self.reranker.rerank(list(documents), query, top_n=self.top_n)
        except Exception as exc:
            secret = getattr(getattr(self.reranker, "cohere_api_key", None), "get_secret_value", lambda: "")()
            msg = redact(f"{type(exc).__name__}: {exc}", [secret])
            log.error(kv(event="rerank_failed", error=msg))
            raise RerankError(f"Cohere rerank failed ({msg})") from None
        kept: list[Document] = []
        for r in sorted(results, key=lambda r: r["relevance_score"], reverse=True):
            idx, score = r["index"], float(r["relevance_score"])
            if score < self.min_score or not (0 <= idx < len(documents)):
                continue
            src = documents[idx]
            kept.append(
                Document(
                    page_content=src.page_content,
                    metadata={**src.metadata, "relevance_score": score},
                )
            )
        return kept[: self.top_n]


def build_reranker(settings: Settings) -> ThresholdCohereRerank:
    from langchain_cohere import CohereRerank

    return ThresholdCohereRerank(
        reranker=CohereRerank(
            model=settings.rerank_model,
            top_n=settings.rerank_top_n,
            cohere_api_key=settings.cohere_api_key,
        ),
        top_n=settings.rerank_top_n,
        min_score=settings.min_relevance_score,
    )


def build_compression_retriever(
    compressor: ThresholdCohereRerank, base_retriever: BaseRetriever
) -> ContextualCompressionRetriever:
    """LangChain-native composition: base retriever (top-20) -> Cohere rerank compressor (top-5)."""
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
