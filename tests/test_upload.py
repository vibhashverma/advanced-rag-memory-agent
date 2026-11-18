"""Document upload endpoint + real PDF ingestion."""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.deps import get_indexer
from app.core.config import get_settings
from app.ingestion.indexing import ChildStore, Indexer, ParentStore
from app.ingestion.loader import load_document
from app.main import create_app
from tests.conftest import HashEmbeddings


def make_pdf(lines: list[str]) -> bytes:
    """Build a minimal valid single-page PDF with a text layer (no external libs)."""
    text = "BT /F1 12 Tf 50 750 Td 14 TL " + " T* ".join(f"({ln}) Tj" for ln in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(text)} >>\nstream\n{text}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


PDF_LINES = ["Hybrid search merges dense vectors with BM25 keyword scores.", "Reranking keeps the five best passages."]


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    indexer = Indexer(
        QdrantClient(":memory:"), HashEmbeddings(), "t",
        ParentStore(tmp_path / "parents"), ChildStore(tmp_path / "children.json"), tmp_path / "manifest.json",
    )
    docs = tmp_path / "docs"
    monkeypatch.setattr("app.api.documents.get_settings", lambda: replace(get_settings(), documents_dir=docs))
    app = create_app()
    app.dependency_overrides[get_indexer] = lambda: (lambda: indexer)
    c = TestClient(app, raise_server_exceptions=False)
    c.indexer, c.docs = indexer, docs
    return c


def test_pdf_text_and_page_metadata_extracted(tmp_path):
    f = tmp_path / "paper.pdf"
    f.write_bytes(make_pdf(PDF_LINES))
    doc_id, pages = load_document(f)
    assert len(pages) == 1 and "Hybrid search merges dense vectors" in pages[0].page_content
    assert pages[0].metadata["page"] == 1 and pages[0].metadata["source"] == "paper.pdf"


def test_upload_pdf_is_ingested(upload_client):
    r = upload_client.post("/api/documents", files={"file": ("paper.pdf", make_pdf(PDF_LINES), "application/pdf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ingested" and body["filename"] == "paper.pdf" and body["children"] >= 1
    assert (upload_client.docs / "paper.pdf").exists()
    kids = upload_client.indexer.child_store.load()
    assert kids and kids[0].metadata["page"] == 1 and kids[0].metadata["source"] == "paper.pdf"


def test_upload_txt_and_reupload_is_skipped(upload_client):
    files = {"file": ("notes.txt", b"RRF merges ranked lists. " * 30, "text/plain")}
    assert upload_client.post("/api/documents", files=files).json()["status"] == "ingested"
    again = upload_client.post("/api/documents", files={"file": ("notes.txt", b"RRF merges ranked lists. " * 30, "text/plain")})
    assert again.status_code == 200 and again.json()["status"] == "skipped"
    assert upload_client.indexer.client.count("t").count == len(upload_client.indexer.child_store.load())


@pytest.mark.parametrize("name,data,status", [
    ("virus.exe", b"MZ", 400),
    ("noext", b"hello", 400),
    ("empty.txt", b"", 400),
    ("blank.txt", b"    \n  ", 422),  # no extractable text
    ("bad.pdf", b"this is not a pdf", 422),
])
def test_invalid_uploads_rejected(upload_client, name, data, status):
    r = upload_client.post("/api/documents", files={"file": (name, data)})
    assert r.status_code == status and "detail" in r.json()
    assert not (upload_client.docs / name).exists()  # failed ingests are not left behind


def test_path_traversal_filename_is_neutralised(upload_client):
    r = upload_client.post("/api/documents", files={"file": ("../../evil.txt", b"safe content here. " * 20)})
    assert r.status_code == 200 and r.json()["filename"] == "evil.txt"
    assert (upload_client.docs / "evil.txt").exists()


def test_missing_file_field_is_422(upload_client):
    assert upload_client.post("/api/documents").status_code == 422


def test_upload_advertised_in_openapi(upload_client):
    assert "/api/documents" in upload_client.get("/openapi.json").json()["paths"]
