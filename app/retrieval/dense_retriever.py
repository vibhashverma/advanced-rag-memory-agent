"""Dense retrieval: embeddings + Qdrant vector search over child chunks."""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.core.config import ConfigurationError, Settings

_NAMESPACE = uuid.UUID("6f1c1a3e-5b7d-4d0e-9a53-2b1e1f0c7a11")


def point_id(child_id: str) -> str:
    """Deterministic Qdrant point id for a child chunk (Qdrant requires UUID/int ids)."""
    return str(uuid.uuid5(_NAMESPACE, child_id))


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {"model": settings.embedding_model, "api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)
    if settings.embedding_provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(model=settings.embedding_model, cohere_api_key=settings.cohere_api_key)
    raise ConfigurationError(f"Unsupported EMBEDDING_PROVIDER {settings.embedding_provider!r}")


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Remote Qdrant if QDRANT_URL is set, otherwise local persistent storage."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


class DenseRetriever:
    """Similarity search over child chunks stored in Qdrant."""

    def __init__(self, client: QdrantClient, embeddings: Embeddings, collection: str, top_k: int = 20) -> None:
        self.client = client
        self.embeddings = embeddings
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        if not self.client.collection_exists(self.collection):
            return []
        vector = self.embeddings.embed_query(query)
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k or self.top_k,
            with_payload=True,
        )
        docs: list[Document] = []
        for p in res.points:
            payload = p.payload or {}
            meta = dict(payload.get("metadata", {}))
            meta["dense_score"] = p.score
            docs.append(Document(page_content=payload.get("page_content", ""), metadata=meta))
        return docs
