"""Dependency wiring. Heavy clients are built lazily (so the server boots and /health works
without credentials) and cached once per process. Tests override these via
`app.dependency_overrides`."""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Callable

from app.agent.generation import Generator, build_chat_model
from app.agent.rag_agent import RagAgent
from app.compression.reranker import build_reranker
from app.core.config import Settings, get_settings
from app.ingestion.indexing import ChildStore, ParentStore
from app.memory.mem0_manager import Mem0Manager, build_mem0_backend
from app.retrieval.dense_retriever import DenseRetriever, build_embeddings, build_qdrant_client
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.parent_retriever import HybridParentRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._chat: ChatService | None = None
        self._memory: Mem0Manager | None = None
        self._llm = None
        self._embeddings = None

    def _secrets(self) -> list[str]:
        s = self.settings
        return [s.openai_api_key, s.anthropic_api_key, s.gemini_api_key, s.cohere_api_key, s.qdrant_api_key, s.mem0_api_key]

    def _shared(self):
        if self._llm is None:
            self._llm = build_chat_model(self.settings)
            self._embeddings = build_embeddings(self.settings)
        return self._llm, self._embeddings

    def memory_manager(self) -> Mem0Manager:
        with self._lock:
            if self._memory is None:
                if not self.settings.mem0_api_key:
                    self.settings.validate()
                llm, emb = (None, None) if self.settings.mem0_api_key else self._shared()
                self._memory = Mem0Manager(build_mem0_backend(self.settings, llm, emb))
            return self._memory

    def chat_service(self) -> ChatService:
        memory = self.memory_manager()  # takes the lock itself; build outside to avoid re-entrancy
        with self._lock:
            if self._chat is None:
                s = self.settings
                s.validate()
                llm, emb = self._shared()
                hybrid = HybridRetriever(
                    DenseRetriever(build_qdrant_client(s), emb, s.qdrant_collection, s.dense_top_k),
                    SparseRetriever(ChildStore(s.children_file).load, s.sparse_top_k),
                    rrf_k=s.rrf_k,
                    dense_top_k=s.dense_top_k,
                    sparse_top_k=s.sparse_top_k,
                    fused_limit=max(s.retrieval_top_k * 2, s.retrieval_top_k),
                )
                retriever = HybridParentRetriever(
                    hybrid=hybrid, parent_store=ParentStore(s.parents_dir), top_k=s.retrieval_top_k
                )
                retrieval = RetrievalService(retriever, build_reranker(s), self._secrets())
                agent = RagAgent(Generator(llm, self._secrets()), max_contexts=s.rerank_top_n)
                self._chat = ChatService(memory, retrieval, agent)
            return self._chat


@lru_cache(maxsize=1)
def get_container() -> Container:
    return Container(get_settings())


# Providers return a *factory* so the handler builds the service only after request validation
# (an invalid body must yield 422 even when credentials are missing).
def get_chat_service() -> Callable[[], ChatService]:
    return get_container().chat_service


def get_memory_manager() -> Callable[[], Mem0Manager]:
    return get_container().memory_manager
