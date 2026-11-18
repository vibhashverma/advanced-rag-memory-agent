"""Long-term semantic memory on Mem0 (open-source SDK or hosted platform), scoped per user_id."""
from __future__ import annotations

import time
from typing import Any, Protocol

from app.core.config import ConfigurationError, Settings
from app.core.logging import get_logger, kv
from app.memory.persona import categorize_fact
from app.schemas.memory import MemoryItem

log = get_logger(__name__)

# Guides Mem0's LLM extraction toward durable facts, away from one-off conversational statements.
EXTRACTION_INSTRUCTIONS = """\
Extract only DURABLE facts and preferences about the user that would help personalize future
conversations: preferred programming languages or tools, response style (e.g. brief, detailed),
output formatting (e.g. code-only), profession/background, and stable interests or constraints.
Write each as a short, self-contained statement (e.g. "Prefers Python", "Prefers brief responses",
"Prefers code-only responses"). Split combined preferences into separate facts.
Do NOT store: greetings, one-off questions or tasks, the topic of a single request, facts about
documents or the assistant's answers, or anything the user only mentioned in passing.
If there is nothing durable to remember, return no facts."""


class MemoryBackend(Protocol):
    """Normalized backend interface. Every record is {id, memory, user_id, metadata, categories?}."""

    def add(self, text: str, user_id: str, metadata: dict[str, Any]) -> list[dict[str, Any]]: ...
    def search(self, query: str, user_id: str, limit: int) -> list[dict[str, Any]]: ...
    def get_all(self, user_id: str, limit: int) -> list[dict[str, Any]]: ...
    def get(self, memory_id: str) -> dict[str, Any] | None: ...
    def update(self, memory_id: str, text: str) -> None: ...
    def delete(self, memory_id: str) -> None: ...


def _results(payload: Any) -> list[dict[str, Any]]:
    """Mem0 returns {'results': [...]} in current SDKs and a bare list in older ones."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        return list(payload.get("results", []))
    return list(payload)


class Mem0OSSBackend:
    """Open-source `mem0.Memory`. Extraction LLM/embedder are the app's configured LangChain models."""

    def __init__(self, memory: Any) -> None:
        self._m = memory

    def add(self, text: str, user_id: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return _results(self._m.add([{"role": "user", "content": text}], user_id=user_id, metadata=metadata))

    def search(self, query: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        return _results(self._m.search(query, filters={"user_id": user_id}, top_k=limit))

    def get_all(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        return _results(self._m.get_all(filters={"user_id": user_id}, top_k=limit))

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return self._m.get(memory_id)

    def update(self, memory_id: str, text: str) -> None:
        self._m.update(memory_id, text)

    def delete(self, memory_id: str) -> None:
        self._m.delete(memory_id)


class Mem0HostedBackend:
    """Hosted Mem0 platform (`mem0.MemoryClient`, selected when MEM0_API_KEY is set)."""

    def __init__(self, client: Any) -> None:
        self._c = client

    EVENT_TIMEOUT_S = 20.0
    EVENT_POLL_S = 0.5

    def add(self, text: str, user_id: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        resp = self._c.add(
            [{"role": "user", "content": text}],
            user_id=user_id,
            metadata=metadata,
            custom_instructions=EXTRACTION_INSTRUCTIONS,
        )
        # The hosted platform extracts asynchronously (status PENDING + event_id). Wait for the
        # event to finish so the facts are stored before this request returns and a new session
        # started right away already sees them.
        if isinstance(resp, dict) and resp.get("event_id") and "results" not in resp:
            return self._await_event(str(resp["event_id"]), user_id)
        return _results(resp)

    def _await_event(self, event_id: str, user_id: str) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.EVENT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                ev = self._c.client.get(f"/v1/event/{event_id}/").json()
            except Exception as exc:
                log.warning(kv(event="mem0_event_poll_failed", error=type(exc).__name__))
                return []
            status = str(ev.get("status", "")).upper()
            if status not in ("PENDING", "RUNNING", "QUEUED"):
                if status in ("FAILED", "ERROR"):
                    raise RuntimeError(f"Mem0 event failed: {ev.get('error')}")
                return [
                    {
                        "id": r.get("id"),
                        "memory": (r.get("data") or {}).get("memory", ""),
                        "user_id": r.get("user_id", user_id),
                        "event": r.get("event", "ADD"),
                    }
                    for r in (ev.get("results") or [])
                ]
            time.sleep(self.EVENT_POLL_S)
        log.warning(kv(event="mem0_event_timeout", event_id=event_id))
        return []

    def search(self, query: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        return _results(self._c.search(query, filters={"user_id": user_id}, top_k=limit))

    def get_all(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        return _results(self._c.get_all(filters={"user_id": user_id}, top_k=limit))

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return self._c.get(memory_id)

    def update(self, memory_id: str, text: str) -> None:
        self._c.update(memory_id, text=text)

    def delete(self, memory_id: str) -> None:
        self._c.delete(memory_id)


_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-v4.0": 1536,
    "gemini-embedding-001": 3072,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def build_mem0_backend(settings: Settings, llm: Any, embeddings: Any) -> MemoryBackend:
    """Hosted client if MEM0_API_KEY is set; otherwise OSS Mem0 persisted in a local Qdrant path."""
    if settings.mem0_api_key:
        from mem0 import MemoryClient

        return Mem0HostedBackend(MemoryClient(api_key=settings.mem0_api_key))

    from mem0 import Memory

    dims = _KNOWN_DIMS.get(settings.embedding_model) or len(embeddings.embed_query("dimension probe"))
    settings.mem0_path.mkdir(parents=True, exist_ok=True)
    config = {
        "llm": {"provider": "langchain", "config": {"model": llm}},
        "embedder": {"provider": "langchain", "config": {"model": embeddings}},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "user_memories",
                "path": str(settings.mem0_path / "qdrant"),
                "embedding_model_dims": dims,
                "on_disk": True,
            },
        },
        "history_db_path": str(settings.mem0_path / "history.db"),
        "custom_instructions": EXTRACTION_INSTRUCTIONS,
    }
    try:
        return Mem0OSSBackend(Memory.from_config(config))
    except Exception as exc:
        raise ConfigurationError(f"Could not initialise Mem0: {type(exc).__name__}: {exc}") from None


class MemoryError_(RuntimeError):
    """Memory backend failure (message safe to show)."""


class MemoryNotFound(LookupError):
    """Memory id does not exist for this user (also raised for other users' ids: no existence leak)."""


class Mem0Manager:
    """Per-user memory operations. Retrieval never depends on session_id, only on user_id."""

    def __init__(self, backend: MemoryBackend, max_memories: int = 100) -> None:
        self.backend = backend
        self.max_memories = max_memories

    # -- helpers -----------------------------------------------------------------
    @staticmethod
    def _require_user(user_id: str) -> str:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        return user_id.strip()

    @staticmethod
    def _to_item(rec: dict[str, Any]) -> MemoryItem:
        fact = rec.get("memory") or rec.get("text") or ""
        meta = rec.get("metadata") or {}
        cats = rec.get("categories") or []
        category = (meta.get("category") if isinstance(meta, dict) else None) or None
        if not category and cats and cats[0] in {"programming_language", "response_style", "formatting"}:
            category = cats[0]
        return MemoryItem(id=str(rec["id"]) if rec.get("id") is not None else None, fact=fact, category=category or categorize_fact(fact))

    def _own(self, recs: list[dict[str, Any]], user_id: str) -> list[dict[str, Any]]:
        """Defence in depth: drop any record the backend tags with a different user."""
        return [r for r in recs if r.get("user_id") in (None, user_id)]

    # -- operations --------------------------------------------------------------
    def add_memory(self, user_id: str, text: str, session_id: str | None = None) -> list[MemoryItem]:
        """Let Mem0 extract durable facts from `text` and persist them for this user."""
        user_id = self._require_user(user_id)
        if not text or not text.strip():
            return []
        meta: dict[str, Any] = {"session_id": session_id} if session_id else {}
        try:
            recs = self.backend.add(text, user_id, meta)
        except Exception as exc:
            log.error(kv(event="memory_add_failed", user_id=user_id, error=type(exc).__name__))
            raise MemoryError_(f"Memory add failed: {type(exc).__name__}") from exc
        # Only newly stored/updated facts are returned (Mem0 emits ADD/UPDATE/DELETE/NONE events).
        recs = [r for r in recs if r.get("event", "ADD") in ("ADD", "UPDATE")]
        return [self._to_item(r) for r in self._own(recs, user_id)]

    def search_memory(self, user_id: str, query: str, limit: int = 5) -> list[MemoryItem]:
        user_id = self._require_user(user_id)
        try:
            recs = self.backend.search(query, user_id, limit)
        except Exception as exc:
            raise MemoryError_(f"Memory search failed: {type(exc).__name__}") from exc
        return [self._to_item(r) for r in self._own(recs, user_id)]

    def get_all_memories(self, user_id: str) -> list[MemoryItem]:
        user_id = self._require_user(user_id)
        try:
            recs = self.backend.get_all(user_id, self.max_memories)
        except Exception as exc:
            raise MemoryError_(f"Memory retrieval failed: {type(exc).__name__}") from exc
        return [self._to_item(r) for r in self._own(recs, user_id)]

    def _assert_owner(self, user_id: str, memory_id: str) -> None:
        rec = self.backend.get(memory_id)
        if not rec or rec.get("user_id") not in (user_id,):
            raise MemoryNotFound(memory_id)

    def update_memory(self, user_id: str, memory_id: str, text: str) -> None:
        user_id = self._require_user(user_id)
        self._assert_owner(user_id, memory_id)
        self.backend.update(memory_id, text)

    def delete_memory(self, user_id: str, memory_id: str) -> None:
        user_id = self._require_user(user_id)
        self._assert_owner(user_id, memory_id)
        self.backend.delete(memory_id)
