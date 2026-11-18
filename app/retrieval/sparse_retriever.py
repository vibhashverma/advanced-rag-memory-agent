"""Sparse (lexical) retrieval using BM25 over child chunks."""
from __future__ import annotations

import re
from typing import Callable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[-.][A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps hyphenated/dotted terms (e.g. 'rerank-english-v3.0') whole
    and also emits their parts so exact terminology and partial matches both work."""
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        parts = re.split(r"[-.]", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


class SparseRetriever:
    """BM25 keyword retriever. The corpus is supplied lazily so newly ingested chunks are picked up;
    the index is rebuilt only when the corpus changes."""

    def __init__(self, corpus_loader: Callable[[], list[Document]], top_k: int = 20) -> None:
        self._load = corpus_loader
        self.top_k = top_k
        self._signature: tuple | None = None
        self._docs: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def _ensure_index(self) -> None:
        docs = self._load()
        signature = tuple(d.metadata.get("child_id") for d in docs)
        if signature == self._signature:
            return
        self._docs = docs
        self._signature = signature
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        self._ensure_index()
        if not self._bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(self._docs)), key=lambda i: scores[i], reverse=True)
        out: list[Document] = []
        for i in ranked[: top_k or self.top_k]:
            if scores[i] <= 0:  # no lexical overlap at all
                break
            d = self._docs[i]
            out.append(Document(page_content=d.page_content, metadata={**d.metadata, "sparse_score": float(scores[i])}))
        return out
