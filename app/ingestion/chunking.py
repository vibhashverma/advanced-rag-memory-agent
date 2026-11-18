"""Parent/child chunking with deterministic IDs and a preserved parent<->child mapping."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_parent_child(
    pages: list[Document],
    document_id: str,
    *,
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
    child_overlap: int = 20,
) -> tuple[list[Document], list[Document]]:
    """Split pages into ~parent_chunk_size parents, then each parent into ~child_chunk_size children.

    IDs are deterministic (`<doc>-p<N>`, `<doc>-p<N>-c<M>`) so re-ingesting identical content
    produces identical records instead of duplicates.
    """
    if child_chunk_size >= parent_chunk_size:
        raise ValueError("child_chunk_size must be smaller than parent_chunk_size")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=min(child_overlap, child_chunk_size // 4)
    )
    parents: list[Document] = []
    children: list[Document] = []
    parent_counter = 0
    for page in pages:
        for parent_text in parent_splitter.split_text(page.page_content):
            if not parent_text.strip():
                continue
            parent_id = f"{document_id}-p{parent_counter}"
            parent_counter += 1
            base = {
                "document_id": document_id,
                "parent_id": parent_id,
                "source": page.metadata.get("source"),
                "page": page.metadata.get("page"),
            }
            parents.append(Document(page_content=parent_text, metadata=dict(base)))
            for j, child_text in enumerate(child_splitter.split_text(parent_text)):
                if not child_text.strip():
                    continue
                children.append(
                    Document(page_content=child_text, metadata={**base, "child_id": f"{parent_id}-c{j}"})
                )
    return parents, children
