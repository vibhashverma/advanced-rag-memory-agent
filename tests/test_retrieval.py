"""TEST 3: parent-document retrieval, plus chunking / ingestion / sparse-dense retrievers."""
from pathlib import Path

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.ingestion.chunking import split_parent_child
from app.ingestion.indexing import ChildStore, Indexer, ParentStore
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.parent_retriever import HybridParentRetriever, children_to_parents
from app.retrieval.sparse_retriever import SparseRetriever
from tests.conftest import FakeParentStore, HashEmbeddings, StaticRetriever, make_child, make_parent


# ------------------------------------------------------------ parent mapping
def test_children_map_to_correct_parent_and_return_full_context():
    store = FakeParentStore([make_parent(0), make_parent(1)])
    res = children_to_parents([make_child(10, parent=1), make_child(11, parent=0)], store)
    assert [p.metadata["parent_id"] for p in res.parents] == ["d1-p1", "d1-p0"]
    assert res.parents[0].page_content == "PARENT 1 full context"  # parent text, not child text
    assert res.parents[0].metadata["source"] == "guide.txt"  # source preserved
    assert res.parents[0].metadata["matched_child_ids"] == ["d1-p1-c10"]


def test_duplicate_parents_removed_and_provenance_merged():
    store = FakeParentStore([make_parent(0), make_parent(1)])
    children = [make_child(1, 0), make_child(2, 1), make_child(3, 0), make_child(4, 0)]
    res = children_to_parents(children, store)
    assert [p.metadata["parent_id"] for p in res.parents] == ["d1-p0", "d1-p1"]
    assert res.parents[0].metadata["matched_child_ids"] == ["d1-p0-c1", "d1-p0-c3", "d1-p0-c4"]


def test_missing_parent_is_handled_not_dropped():
    store = FakeParentStore([make_parent(0)])
    res = children_to_parents([make_child(1, 0), make_child(2, 9)], store)
    assert res.missing_parent_ids == ["d1-p9"]
    fallback = res.parents[1]
    assert fallback.metadata["parent_missing"] is True and fallback.page_content == "child text 2"


def test_missing_parent_children_not_duplicated():
    res = children_to_parents([make_child(1, 9), make_child(2, 9)], FakeParentStore([]))
    assert len(res.parents) == 1


def test_limit_and_empty():
    store = FakeParentStore([make_parent(i) for i in range(5)])
    assert len(children_to_parents([make_child(i) for i in range(5)], store, limit=3).parents) == 3
    assert children_to_parents([], store).parents == []


def test_hybrid_parent_retriever_end_to_end_with_mocks():
    kids = [make_child(i, parent=i // 2) for i in range(6)]  # 3 parents, 2 children each
    hybrid = HybridRetriever(StaticRetriever(kids[:4]), StaticRetriever(kids[2:]))
    r = HybridParentRetriever(hybrid=hybrid, parent_store=FakeParentStore([make_parent(i) for i in range(3)]), top_k=20)
    parents, stats = r.retrieve_with_stats("q")
    assert len(parents) == 3 and len({p.metadata["parent_id"] for p in parents}) == 3
    assert (stats.dense_count, stats.sparse_count, stats.parent_candidates, stats.missing_parents) == (4, 4, 3, 0)
    assert len(r.invoke("q")) == 3  # also usable as a LangChain retriever


# ------------------------------------------------------------ chunking
def test_chunking_sizes_ids_and_mapping():
    text = " ".join(f"Sentence number {i} talks about retrieval." for i in range(120))
    pages = [Document(page_content=text, metadata={"source": "a.txt", "page": 2})]
    parents, children = split_parent_child(pages, "doc1", parent_chunk_size=1000, child_chunk_size=200)
    assert len(parents) > 1 and len(children) > len(parents)
    assert all(len(p.page_content) <= 1000 for p in parents)
    assert all(len(c.page_content) <= 200 for c in children)
    pids = {p.metadata["parent_id"] for p in parents}
    for c in children:
        m = c.metadata
        assert m["parent_id"] in pids and m["child_id"].startswith(m["parent_id"] + "-c")
        assert m["document_id"] == "doc1" and m["source"] == "a.txt" and m["page"] == 2
    assert len({c.metadata["child_id"] for c in children}) == len(children)


def test_chunk_size_validation():
    with pytest.raises(ValueError):
        split_parent_child([], "d", parent_chunk_size=100, child_chunk_size=100)


# ------------------------------------------------------------ ingestion (in-memory Qdrant)
@pytest.fixture
def indexer(tmp_path):
    return Indexer(
        QdrantClient(":memory:"), HashEmbeddings(), "test",
        ParentStore(tmp_path / "parents"), ChildStore(tmp_path / "children.json"), tmp_path / "manifest.json",
    )


def write_doc(tmp_path: Path, name="notes.txt", body=None) -> Path:
    p = tmp_path / name
    p.write_text(body or ("Hybrid search merges BM25 keyword scores with dense vectors. " * 60), encoding="utf-8")
    return p


def test_ingestion_is_idempotent(indexer, tmp_path):
    f = write_doc(tmp_path)
    first = indexer.ingest_file(f)
    n_points = indexer.client.count("test").count
    second = indexer.ingest_file(f)
    forced = indexer.ingest_file(f, force=True)
    assert first.status == "ingested" and second.status == "skipped" and forced.status == "ingested"
    assert indexer.client.count("test").count == n_points == first.children
    assert len(indexer.child_store.load()) == first.children


def test_changed_file_replaces_old_records(indexer, tmp_path):
    f = write_doc(tmp_path)
    indexer.ingest_file(f)
    f.write_text("Completely different content about parent document retrieval. " * 40, encoding="utf-8")
    res = indexer.ingest_file(f)
    assert res.status == "ingested"
    assert indexer.client.count("test").count == res.children  # old vectors removed
    assert {c.metadata["document_id"] for c in indexer.child_store.load()} == {res.document_id}


def test_ingested_children_resolve_to_stored_parents(indexer, tmp_path):
    indexer.ingest_file(write_doc(tmp_path))
    children = indexer.child_store.load()
    parents = indexer.parent_store.get_many([c.metadata["parent_id"] for c in children])
    assert all(c.metadata["parent_id"] in parents for c in children)


def test_unsupported_and_empty_files_reported_not_raised(indexer, tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("hi")
    empty = write_doc(tmp_path, "empty.txt", body="   ")
    assert indexer.ingest_file(bad).status == "failed"
    assert indexer.ingest_file(empty).status == "failed"


# ------------------------------------------------------------ dense & sparse
def test_dense_retriever_returns_scored_children(indexer, tmp_path):
    indexer.ingest_file(write_doc(tmp_path))
    dense = DenseRetriever(indexer.client, HashEmbeddings(), "test", top_k=5)
    out = dense.retrieve("hybrid search")
    assert 0 < len(out) <= 5 and "dense_score" in out[0].metadata and out[0].metadata["child_id"]


def test_dense_retriever_empty_when_no_collection():
    assert DenseRetriever(QdrantClient(":memory:"), HashEmbeddings(), "none").retrieve("q") == []


def test_sparse_retriever_exact_terminology_match():
    corpus = [
        Document(page_content="Cohere rerank-english-v3.0 scores relevance", metadata={"child_id": "a"}),
        Document(page_content="Qdrant stores dense vectors on disk", metadata={"child_id": "b"}),
        Document(page_content="Cats sleep a lot", metadata={"child_id": "c"}),
    ]
    sparse = SparseRetriever(lambda: corpus, top_k=5)
    assert sparse.retrieve("rerank-english-v3.0")[0].metadata["child_id"] == "a"
    assert [d.metadata["child_id"] for d in sparse.retrieve("qdrant")] == ["b"]
    assert sparse.retrieve("zzzzunknown") == [] and sparse.retrieve("   ") == []


def test_sparse_retriever_picks_up_new_chunks():
    corpus: list[Document] = []
    sparse = SparseRetriever(lambda: corpus)
    assert sparse.retrieve("alpha") == []
    corpus.append(Document(page_content="alpha beta", metadata={"child_id": "x"}))
    assert len(sparse.retrieve("alpha")) == 1


def test_large_document_is_embedded_and_upserted_in_batches(indexer, tmp_path):
    indexer.BATCH_SIZE = 5
    calls = []
    real = indexer.embeddings.embed_documents
    indexer.embeddings.embed_documents = lambda texts: (calls.append(len(texts)), real(texts))[1]
    res = indexer.ingest_file(write_doc(tmp_path, body="Article 21 protects life and personal liberty. " * 200))
    assert res.status == "ingested" and res.children > 15
    assert len(calls) > 1 and max(calls) <= 5 and sum(calls) == res.children
    assert indexer.client.count("test").count == res.children
