"""Shared test doubles. External services (Cohere, LLM, embeddings, Mem0's extraction LLM) are
replaced here; all application code under test is the real implementation."""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.agent.generation import Generator
from app.agent.rag_agent import RagAgent
from app.api.deps import get_chat_service, get_memory_manager
from app.compression.reranker import ThresholdCohereRerank
from app.ingestion.indexing import ParentStore
from app.main import create_app
from app.memory.mem0_manager import Mem0Manager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.parent_retriever import HybridParentRetriever
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService


# ---------------------------------------------------------------- documents
def make_child(i: int, parent: int | None = None, doc: str = "d1") -> Document:
    parent = i if parent is None else parent
    return Document(
        page_content=f"child text {i}",
        metadata={
            "document_id": doc,
            "parent_id": f"{doc}-p{parent}",
            "child_id": f"{doc}-p{parent}-c{i}",
            "source": "guide.txt",
            "page": None,
        },
    )


def make_parent(p: int, doc: str = "d1") -> Document:
    return Document(
        page_content=f"PARENT {p} full context",
        metadata={"document_id": doc, "parent_id": f"{doc}-p{p}", "source": "guide.txt", "page": None},
    )


class FakeParentStore:
    def __init__(self, parents: list[Document]) -> None:
        self._p = {d.metadata["parent_id"]: d for d in parents}

    def get_many(self, ids: list[str]) -> dict[str, Document]:
        return {i: self._p[i] for i in ids if i in self._p}


class StaticRetriever:
    """Chunk retriever returning a fixed list (mock of dense/sparse retrieval)."""

    def __init__(self, docs: list[Document], delay: float = 0.0) -> None:
        self.docs, self.delay, self.calls = docs, delay, []

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        self.calls.append(query)
        time.sleep(self.delay)
        return list(self.docs[:top_k] if top_k else self.docs)


class HashEmbeddings(Embeddings):
    """Deterministic fake embeddings for ingestion tests."""

    def _v(self, t: str) -> list[float]:
        return [b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._v(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._v(text)


# ---------------------------------------------------------------- cohere
def fake_cohere_reranker(scores: dict[int, float] | Callable[[int], list[tuple[int, float]]] | None = None):
    """Mock of langchain_cohere.CohereRerank: .rerank(documents, query, top_n=) -> [{index, relevance_score}]."""
    calls: list[dict] = []

    def rerank(documents, query, top_n=None, **_):
        calls.append({"n_docs": len(documents), "query": query, "top_n": top_n})
        if callable(scores):
            return [{"index": i, "relevance_score": s} for i, s in scores(len(documents))]
        mapping = scores if scores is not None else {i: 1.0 - i * 0.01 for i in range(len(documents))}
        return [{"index": i, "relevance_score": s} for i, s in mapping.items() if i < len(documents)]

    return SimpleNamespace(rerank=rerank, calls=calls, cohere_api_key=None)


# ---------------------------------------------------------------- memory
_PREF = re.compile(r"prefer\w*\s+(?P<what>[^.!?]+)", re.I)


def keyword_extractor(text: str) -> list[str]:
    """Stands in for Mem0's LLM fact extraction: durable facts only, never greetings/questions."""
    facts: list[str] = []
    m = re.search(r"work in (\w+)", text, re.I)
    if m:
        facts.append(f"Prefers {m.group(1).capitalize()}")
    m = _PREF.search(text)
    if m:
        for part in re.split(r",| and ", m.group("what")):
            part = part.strip()
            if part:
                facts.append(f"Prefers {part}")
    return facts


class FakeMemoryBackend:
    """In-memory MemoryBackend with strict per-user partitioning (like Mem0 user_id filters)."""

    def __init__(self, extractor: Callable[[str], list[str]] = keyword_extractor) -> None:
        self.extractor = extractor
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("memory backend down")

    def add(self, text, user_id, metadata):
        self._check()
        out = []
        for fact in self.extractor(text):
            if any(r["user_id"] == user_id and r["memory"] == fact for r in self.rows.values()):
                continue
            rec = {"id": str(uuid.uuid4()), "memory": fact, "user_id": user_id, "metadata": dict(metadata), "event": "ADD"}
            self.rows[rec["id"]] = rec
            out.append(dict(rec))
        return out

    def search(self, query, user_id, limit):
        self._check()
        words = set(re.findall(r"\w+", query.lower()))
        hits = [r for r in self.rows.values() if r["user_id"] == user_id and words & set(re.findall(r"\w+", r["memory"].lower()))]
        return [dict(r) for r in hits[:limit]]

    def get_all(self, user_id, limit):
        self._check()
        return [dict(r) for r in self.rows.values() if r["user_id"] == user_id][:limit]

    def get(self, memory_id):
        return self.rows.get(memory_id)

    def update(self, memory_id, text):
        self.rows[memory_id]["memory"] = text

    def delete(self, memory_id):
        self.rows.pop(memory_id, None)


@pytest.fixture
def memory_backend() -> FakeMemoryBackend:
    return FakeMemoryBackend()


@pytest.fixture
def memory_manager(memory_backend) -> Mem0Manager:
    return Mem0Manager(memory_backend)


# ---------------------------------------------------------------- LLM
class FakeLLM:
    """Records every prompt it receives and answers according to the persona in the system prompt."""

    def __init__(self, delay: float = 0.002, fail: Exception | None = None) -> None:
        self.delay, self.fail, self.calls = delay, fail, []

    def invoke(self, messages):
        time.sleep(self.delay)
        if self.fail:
            raise self.fail
        system, user = messages[0].content, messages[1].content
        self.calls.append({"system": system, "user": user})
        if "code-only" in system.lower():
            text = "```python\ndef merge_sort(a): ...\n```"
        else:
            text = "Here is a detailed explanation with context [1]."
        return SimpleNamespace(content=text)


# ---------------------------------------------------------------- pipeline
@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def cohere_mock():
    return fake_cohere_reranker()


@pytest.fixture
def chat_service(memory_manager, llm, cohere_mock) -> ChatService:
    """Real hybrid/parent/rerank/agent code over mocked dense, sparse, Cohere and LLM."""
    n = 24
    children = [make_child(i) for i in range(n)]
    dense = StaticRetriever(children[:20], delay=0.002)
    sparse = StaticRetriever(list(reversed(children[4:24])), delay=0.002)
    hybrid = HybridRetriever(dense, sparse, rrf_k=60, dense_top_k=20, sparse_top_k=20, fused_limit=40)
    store = FakeParentStore([make_parent(i) for i in range(n)])
    retriever = HybridParentRetriever(hybrid=hybrid, parent_store=store, top_k=20)
    compressor = ThresholdCohereRerank(reranker=cohere_mock, top_n=5, min_score=0.5)
    return ChatService(
        memory_manager,
        RetrievalService(retriever, compressor),
        RagAgent(Generator(llm), max_contexts=5),
    )


@pytest.fixture
def client(chat_service, memory_manager):
    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: (lambda: chat_service)
    app.dependency_overrides[get_memory_manager] = lambda: (lambda: memory_manager)
    return TestClient(app, raise_server_exceptions=False)
