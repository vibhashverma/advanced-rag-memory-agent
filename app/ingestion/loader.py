"""Load local PDF/TXT files into LangChain Documents with metadata."""
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class UnsupportedFileError(ValueError):
    pass


def file_hash(path: Path) -> str:
    """Content hash -> stable document_id, so re-ingesting a file is idempotent."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_document(path: Path) -> tuple[str, list[Document]]:
    """Return (document_id, pages). Each page Document carries source/page metadata."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}")
    document_id = file_hash(path)[:16]
    if ext == ".pdf":
        pages = PyPDFLoader(str(path)).load()
    else:
        pages = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    out: list[Document] = []
    for i, page in enumerate(pages):
        raw_page = page.metadata.get("page")
        out.append(
            Document(
                page_content=page.page_content,
                metadata={
                    "document_id": document_id,
                    "source": path.name,
                    "file_type": ext.lstrip("."),
                    # pypdf pages are 0-indexed; expose 1-indexed. TXT has no pages.
                    "page": (int(raw_page) + 1) if raw_page is not None else None,
                    "page_index": i,
                },
            )
        )
    return document_id, out
