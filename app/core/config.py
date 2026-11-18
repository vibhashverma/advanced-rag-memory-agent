"""Application configuration loaded from environment variables (.env via python-dotenv)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_LLM_PROVIDERS = {"openai", "anthropic", "cohere", "gemini"}
SUPPORTED_EMBEDDING_PROVIDERS = {"openai", "cohere", "gemini", "huggingface"}


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


def _path(name: str, default: str) -> Path:
    p = Path(_str(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    anthropic_api_key: str
    gemini_api_key: str
    cohere_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_path: Path
    qdrant_collection: str
    mem0_api_key: str
    mem0_path: Path
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    rerank_model: str
    retrieval_top_k: int
    rerank_top_n: int
    min_relevance_score: float
    child_chunk_size: int
    parent_chunk_size: int
    rrf_k: int
    dense_top_k: int
    sparse_top_k: int
    data_dir: Path
    documents_dir: Path
    log_level: str

    # -- derived paths -------------------------------------------------
    @property
    def parents_dir(self) -> Path:
        return self.data_dir / "parents"

    @property
    def children_file(self) -> Path:
        return self.data_dir / "children.json"

    @property
    def manifest_file(self) -> Path:
        return self.data_dir / "ingest_manifest.json"

    # -- validation ----------------------------------------------------
    def validate(self, *, for_generation: bool = True) -> None:
        """Raise ConfigurationError listing every missing/invalid setting."""
        problems: list[str] = []
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            problems.append(f"LLM_PROVIDER must be one of {sorted(SUPPORTED_LLM_PROVIDERS)}")
        if self.embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            problems.append(
                f"EMBEDDING_PROVIDER must be one of {sorted(SUPPORTED_EMBEDDING_PROVIDERS)}"
            )
        if not self.cohere_api_key:
            problems.append("COHERE_API_KEY is required (Cohere Rerank)")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            problems.append("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is required for EMBEDDING_PROVIDER=gemini")
        if for_generation:
            keys = {
                "openai": self.openai_api_key,
                "anthropic": self.anthropic_api_key,
                "gemini": self.gemini_api_key,
                "cohere": self.cohere_api_key,
            }
            if self.llm_provider in keys and not keys[self.llm_provider]:
                problems.append(
                    f"{self.llm_provider.upper()}_API_KEY is required for LLM_PROVIDER={self.llm_provider}"
                )
            if not self.llm_model:
                problems.append("LLM_MODEL is required")
        if self.rerank_top_n > self.retrieval_top_k:
            problems.append("RERANK_TOP_N must be <= RETRIEVAL_TOP_K")
        if self.child_chunk_size >= self.parent_chunk_size:
            problems.append("CHILD_CHUNK_SIZE must be smaller than PARENT_CHUNK_SIZE")
        if problems:
            raise ConfigurationError(
                "Invalid configuration (see .env.example): " + "; ".join(problems)
            )


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        openai_api_key=_str("OPENAI_API_KEY"),
        openai_base_url=_str("OPENAI_BASE_URL"),
        anthropic_api_key=_str("ANTHROPIC_API_KEY"),
        gemini_api_key=_str("GEMINI_API_KEY") or _str("GOOGLE_API_KEY"),
        cohere_api_key=_str("COHERE_API_KEY"),
        qdrant_url=_str("QDRANT_URL"),
        qdrant_api_key=_str("QDRANT_API_KEY"),
        qdrant_path=_path("QDRANT_PATH", "data/qdrant_storage"),
        qdrant_collection=_str("QDRANT_COLLECTION", "rag_children"),
        mem0_api_key=_str("MEM0_API_KEY"),
        mem0_path=_path("MEM0_PATH", "data/mem0"),
        llm_provider=_str("LLM_PROVIDER", "openai").lower(),
        llm_model=_str("LLM_MODEL", "gpt-4o-mini"),
        embedding_provider=_str("EMBEDDING_PROVIDER", "openai").lower(),
        embedding_model=_str("EMBEDDING_MODEL", "text-embedding-3-small"),
        rerank_model=_str("RERANK_MODEL", "rerank-english-v3.0"),
        retrieval_top_k=_int("RETRIEVAL_TOP_K", 20),
        rerank_top_n=_int("RERANK_TOP_N", 5),
        min_relevance_score=_float("MIN_RELEVANCE_SCORE", 0.1),
        child_chunk_size=_int("CHILD_CHUNK_SIZE", 200),
        parent_chunk_size=_int("PARENT_CHUNK_SIZE", 1000),
        rrf_k=_int("RRF_K", 60),
        dense_top_k=_int("DENSE_TOP_K", 20),
        sparse_top_k=_int("SPARSE_TOP_K", 20),
        data_dir=_path("DATA_DIR", "data"),
        documents_dir=_path("DOCUMENTS_DIR", "data/documents"),
        log_level=_str("LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
