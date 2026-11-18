"""Parent-document retrieval: map matched child chunks to their full parent context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.core.logging import get_logger, kv
from app.retrieval.hybrid_retriever import HybridResult, HybridRetriever

log = get_logger(__name__)


class ParentLookup(Protocol):
    def get_many(self, parent_ids: list[str]) -> dict[str, Document]: ...


@dataclass
class ParentResult:
    parents: list[Document]
    missing_parent_ids: list[str] = field(default_factory=list)


def children_to_parents(
    children: list[Document], store: ParentLookup, limit: int | None = None
) -> ParentResult:
    """Replace ranked child chunks with their parents.

    - Order follows the best-ranked child of each parent.
    - Each parent appears once, however many of its children matched.
    - Children whose parent cannot be found are kept as-is (flagged `parent_missing`) so
      evidence is not silently lost; their ids are reported in `missing_parent_ids`.
    """
    parent_ids = list(dict.fromkeys(c.metadata["parent_id"] for c in children if c.metadata.get("parent_id")))
    found = store.get_many(parent_ids)
    out: list[Document] = []
    emitted: set[str] = set()
    missing: list[str] = []
    for child in children:
        pid = child.metadata.get("parent_id")
        child_id = child.metadata.get("child_id")
        if pid in found:
            if pid in emitted:
                # extend provenance of the already-emitted parent
                for p in out:
                    if p.metadata.get("parent_id") == pid:
                        p.metadata["matched_child_ids"].append(child_id)
                        break
                continue
            emitted.add(pid)
            parent = found[pid]
            out.append(
                Document(
                    page_content=parent.page_content,
                    metadata={
                        **parent.metadata,
                        "matched_child_ids": [child_id],
                        "rrf_score": child.metadata.get("rrf_score"),
                    },
                )
            )
        else:
            key = pid or child_id
            if key in emitted:
                continue
            emitted.add(key)
            missing.append(str(pid))
            log.warning(kv(event="parent_missing", parent_id=pid, child_id=child_id))
            out.append(
                Document(
                    page_content=child.page_content,
                    metadata={**child.metadata, "matched_child_ids": [child_id], "parent_missing": True},
                )
            )
    return ParentResult(parents=out[:limit] if limit else out, missing_parent_ids=missing)


@dataclass
class RetrievalStats:
    dense_count: int
    sparse_count: int
    fused_children: int
    parent_candidates: int
    missing_parents: int


class HybridParentRetriever(BaseRetriever):
    """LangChain retriever: dense+sparse -> RRF -> children -> parents (top-k candidates)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid: Any
    parent_store: Any  # ParentLookup
    top_k: int = 20

    def retrieve_with_stats(self, query: str) -> tuple[list[Document], RetrievalStats]:
        res: HybridResult = self.hybrid.retrieve(query)
        parents = children_to_parents(res.children, self.parent_store, limit=self.top_k)
        stats = RetrievalStats(
            dense_count=res.dense_count,
            sparse_count=res.sparse_count,
            fused_children=res.fused_count,
            parent_candidates=len(parents.parents),
            missing_parents=len(parents.missing_parent_ids),
        )
        return parents.parents, stats

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        return self.retrieve_with_stats(query)[0]
