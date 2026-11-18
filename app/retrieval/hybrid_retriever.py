"""Hybrid retrieval: dense + sparse child-chunk search fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from app.retrieval.rrf import reciprocal_rank_fusion


class ChunkRetriever(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]: ...


@dataclass
class HybridResult:
    children: list[Document]  # fused, RRF-ranked child chunks
    dense_count: int
    sparse_count: int
    fused_count: int


class HybridRetriever:
    def __init__(
        self,
        dense: ChunkRetriever,
        sparse: ChunkRetriever,
        *,
        rrf_k: int = 60,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        fused_limit: int = 40,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fused_limit = fused_limit

    def retrieve(self, query: str) -> HybridResult:
        dense_docs = self.dense.retrieve(query, self.dense_top_k)
        sparse_docs = self.sparse.retrieve(query, self.sparse_top_k)
        fused = reciprocal_rank_fusion([dense_docs, sparse_docs], k=self.rrf_k, limit=self.fused_limit)
        return HybridResult(
            children=fused,
            dense_count=len(dense_docs),
            sparse_count=len(sparse_docs),
            fused_count=len(fused),
        )
