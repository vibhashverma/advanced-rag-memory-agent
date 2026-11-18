"""Persistent stores (parents, children) and the idempotent indexing pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.core.logging import get_logger, kv
from app.ingestion.chunking import split_parent_child
from app.ingestion.loader import discover_files, load_document
from app.retrieval.dense_retriever import ensure_collection, ensure_document_index, point_id

log = get_logger(__name__)


def _atomic_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class ParentStore:
    """Local document repository: one JSON file per ingested document holding its parents."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _file(self, document_id: str) -> Path:
        return self.directory / f"{document_id}.json"

    def put_document(self, document_id: str, parents: list[Document]) -> None:
        _atomic_write(
            self._file(document_id),
            {p.metadata["parent_id"]: {"text": p.page_content, "metadata": p.metadata} for p in parents},
        )

    def delete_document(self, document_id: str) -> None:
        f = self._file(document_id)
        if f.exists():
            f.unlink()

    def get_many(self, parent_ids: list[str]) -> dict[str, Document]:
        """Return {parent_id: Document}; ids that cannot be resolved are simply absent."""
        found: dict[str, Document] = {}
        cache: dict[str, dict] = {}
        for pid in parent_ids:
            doc_id = pid.rsplit("-p", 1)[0]
            if doc_id not in cache:
                cache[doc_id] = _read_json(self._file(doc_id), {})
            rec = cache[doc_id].get(pid)
            if rec:
                found[pid] = Document(page_content=rec["text"], metadata=dict(rec["metadata"]))
        return found


class ChildStore:
    """All child chunks (text + metadata); the corpus the sparse (BM25) index is built from."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Document]:
        data = _read_json(self.path, {})
        return [Document(page_content=v["text"], metadata=v["metadata"]) for v in data.values()]

    def replace_document(self, document_id: str, children: list[Document]) -> None:
        data = _read_json(self.path, {})
        data = {k: v for k, v in data.items() if v["metadata"].get("document_id") != document_id}
        for c in children:
            data[c.metadata["child_id"]] = {"text": c.page_content, "metadata": c.metadata}
        _atomic_write(self.path, data)


@dataclass
class IngestResult:
    source: str
    document_id: str
    status: str  # "ingested" | "skipped" | "failed"
    parents: int = 0
    children: int = 0
    error: str | None = None


class Indexer:
    """Idempotent ingestion: unchanged files are skipped; changed files replace their old records."""

    BATCH_SIZE = 128

    def __init__(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
        collection: str,
        parent_store: ParentStore,
        child_store: ChildStore,
        manifest_path: Path,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.collection = collection
        self.parent_store = parent_store
        self.child_store = child_store
        self.manifest_path = Path(manifest_path)
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def ingest_directory(self, directory: Path, force: bool = False) -> list[IngestResult]:
        return [self.ingest_file(p, force=force) for p in discover_files(directory)]

    def ingest_file(self, path: Path, force: bool = False) -> IngestResult:
        path = Path(path)
        try:
            document_id, pages = load_document(path)
            manifest = _read_json(self.manifest_path, {})
            if not force and manifest.get(path.name) == document_id:
                return IngestResult(path.name, document_id, "skipped")
            parents, children = split_parent_child(
                pages,
                document_id,
                parent_chunk_size=self.parent_chunk_size,
                child_chunk_size=self.child_chunk_size,
            )
            if not children:
                return IngestResult(path.name, document_id, "failed", error="no extractable text")
            # Remove stale records (an edited file gets a new content hash; same-hash re-ingest replaces).
            old_id = manifest.get(path.name)
            if old_id and old_id != document_id:
                self._delete_vectors(old_id)
                self.parent_store.delete_document(old_id)
                self.child_store.replace_document(old_id, [])
            self._delete_vectors(document_id)
            # Embed + upsert in batches so large documents (hundreds of pages) stay within request
            # size limits and memory bounds.
            for start in range(0, len(children), self.BATCH_SIZE):
                batch = children[start : start + self.BATCH_SIZE]
                vectors = self.embeddings.embed_documents([c.page_content for c in batch])
                ensure_collection(self.client, self.collection, len(vectors[0]))
                self.client.upsert(
                    collection_name=self.collection,
                    points=[
                        models.PointStruct(
                            id=point_id(c.metadata["child_id"]),
                            vector=v,
                            payload={"page_content": c.page_content, "metadata": c.metadata},
                        )
                        for c, v in zip(batch, vectors)
                    ],
                )
            self.parent_store.put_document(document_id, parents)
            self.child_store.replace_document(document_id, children)
            manifest[path.name] = document_id
            _atomic_write(self.manifest_path, manifest)
            log.info(kv(event="ingested", source=path.name, parents=len(parents), children=len(children)))
            return IngestResult(path.name, document_id, "ingested", len(parents), len(children))
        except Exception as exc:  # report per-file failure, keep going
            log.exception(kv(event="ingest_failed", source=path.name))
            return IngestResult(path.name, "", "failed", error=str(exc))

    def _delete_vectors(self, document_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        ensure_document_index(self.client, self.collection)
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
        )
