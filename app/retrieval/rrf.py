"""Reciprocal Rank Fusion: RRF_score(d) = sum_i 1 / (k + rank_i(d))."""
from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.documents import Document

DEFAULT_K = 60


def default_doc_id(doc: Document) -> str:
    return str(doc.metadata.get("child_id") or doc.metadata.get("id") or doc.page_content)


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Document]],
    *,
    k: int = DEFAULT_K,
    limit: int | None = None,
    id_fn: Callable[[Document], str] = default_doc_id,
) -> list[Document]:
    """Fuse ranked lists. Ranks are 1-based. A document present in several lists is merged into
    one result whose `rrf_score` is the sum of its per-list contributions. Ties keep the order of
    first appearance, so output is deterministic."""
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    docs: dict[str, Document] = {}
    ranks: dict[str, dict[int, int]] = {}
    for list_idx, results in enumerate(result_lists):
        seen_in_list: set[str] = set()
        for rank, doc in enumerate(results, start=1):
            doc_id = id_fn(doc)
            if doc_id in seen_in_list:  # a duplicate inside one list must not be double counted
                continue
            seen_in_list.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(doc_id, {})[list_idx] = rank
            if doc_id not in docs:
                docs[doc_id] = doc
                first_seen[doc_id] = len(first_seen)
            else:  # carry per-retriever scores from the other list onto the merged doc
                docs[doc_id] = Document(
                    page_content=docs[doc_id].page_content,
                    metadata={**doc.metadata, **docs[doc_id].metadata},
                )
    ordered = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    fused = [
        Document(
            page_content=docs[d].page_content,
            metadata={**docs[d].metadata, "rrf_score": scores[d], "rrf_ranks": ranks[d]},
        )
        for d in ordered
    ]
    return fused[:limit] if limit is not None else fused
