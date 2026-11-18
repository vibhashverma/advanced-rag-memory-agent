"""Ingest PDF/TXT files into the parent store, child store and Qdrant.

Usage:
    python scripts/ingest_documents.py [PATH ...] [--force]

With no PATH, ingests everything under DOCUMENTS_DIR (default data/documents).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import ConfigurationError, get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.ingestion.indexing import ChildStore, Indexer, ParentStore  # noqa: E402
from app.retrieval.dense_retriever import build_embeddings, build_qdrant_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="files or directories (default: DOCUMENTS_DIR)")
    ap.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    args = ap.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        settings.validate(for_generation=False)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    indexer = Indexer(
        client=build_qdrant_client(settings),
        embeddings=build_embeddings(settings),
        collection=settings.qdrant_collection,
        parent_store=ParentStore(settings.parents_dir),
        child_store=ChildStore(settings.children_file),
        manifest_path=settings.manifest_file,
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
    )
    targets = [Path(p) for p in args.paths] or [settings.documents_dir]
    results = []
    for t in targets:
        if t.is_dir():
            results += indexer.ingest_directory(t, args.force)
        else:
            results.append(indexer.ingest_file(t, args.force))
    if not results:
        print(f"No PDF/TXT files found in {targets}")
    for r in results:
        extra = f"parents={r.parents} children={r.children}" if r.status == "ingested" else (r.error or "")
        print(f"[{r.status}] {r.source} {extra}")
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
