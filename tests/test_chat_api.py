"""TEST 5: FastAPI chat endpoint, memories endpoint, multi-session persona."""
import re

import pytest

from app.agent.generation import GenerationError
from app.core.config import ConfigurationError
from app.api.deps import get_chat_service

VALID = {"query": "Explain hybrid search", "user_id": "usr_99", "session_id": "session_001"}
PERSONA_MSG = "Hello! I work in Python and prefer brief, code-only responses."


def test_valid_request_returns_expected_structure(client):
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "user_id", "session_id", "sources", "telemetry"}
    assert body["user_id"] == "usr_99" and body["session_id"] == "session_001" and body["answer"]
    for s in body["sources"]:
        assert {"source", "page", "parent_id", "relevance_score", "snippet"} <= set(s)


@pytest.mark.parametrize("payload", [
    {},
    {"query": "hi", "user_id": "u"},  # missing session
    {"query": "hi", "session_id": "s"},  # missing user
    {"query": "   ", "user_id": "u", "session_id": "s"},  # blank query
    {"query": "", "user_id": "u", "session_id": "s"},
    {"query": "x" * 4001, "user_id": "u", "session_id": "s"},
    {"query": "hi", "user_id": "bad user!", "session_id": "s"},
    {"query": "hi", "user_id": "u", "session_id": ""},
    {"query": 5, "user_id": "u", "session_id": "s"},
])
def test_invalid_requests_rejected(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_non_json_body_rejected(client):
    assert client.post("/api/chat", content="not json", headers={"content-type": "application/json"}).status_code == 422


def test_pipeline_reduces_20_candidates_to_at_most_5_before_llm(client, llm):
    body = client.post("/api/chat", json=VALID).json()
    t = body["telemetry"]
    assert t["retrieved_candidates"] == 20  # fused+parent candidates handed to the reranker
    assert t["dense_hits"] == 20 and t["sparse_hits"] == 20
    assert 0 < t["compressed_contexts"] <= 5
    assert len(body["sources"]) == t["compressed_contexts"]
    prompt = llm.calls[-1]["user"]
    assert len(re.findall(r"^\[\d+\] \(", prompt, flags=re.M)) <= 5
    assert "PARENT" in prompt  # parent context, not isolated child chunks
    assert "child text" not in prompt
    assert all(s["relevance_score"] >= 0.5 for s in body["sources"])  # threshold enforced


def test_telemetry_block_present(client):
    t = client.post("/api/chat", json=VALID).json()["telemetry"]
    for key in ("retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms", "total_latency_ms"):
        assert isinstance(t[key], (int, float)) and t[key] > 0


def test_multi_session_persona_adaptation(client, llm):
    # Session 1: user states preferences; they are extracted + stored.
    r1 = client.post("/api/chat", json={"query": PERSONA_MSG, "user_id": "usr_99", "session_id": "session_001"})
    assert r1.status_code == 200 and r1.json()["telemetry"]["memories_saved"] >= 2

    mem = client.get("/api/users/usr_99/memories")
    assert mem.status_code == 200
    stored = {(m["fact"], m["category"]) for m in mem.json()["memories"]}
    assert ("Prefers Python", "programming_language") in stored
    assert any(cat == "formatting" for _, cat in stored)

    # Session 2 (new session id): persona is injected from memory without being restated.
    r2 = client.post("/api/chat", json={"query": "Write a quick merge sort routine.", "user_id": "usr_99", "session_id": "session_002"})
    assert r2.status_code == 200
    system = llm.calls[-1]["system"]
    assert "USER PROFILE" in system and "Prefers Python" in system and "code-only" in system.lower()
    assert r2.json()["answer"].startswith("```python")
    assert r2.json()["telemetry"]["memories_used"] >= 3


def test_other_users_do_not_receive_persona(client, llm):
    client.post("/api/chat", json={"query": PERSONA_MSG, "user_id": "usr_99", "session_id": "s1"})
    r = client.post("/api/chat", json={"query": "Write a merge sort", "user_id": "usr_100", "session_id": "s1"})
    assert r.status_code == 200
    assert "USER PROFILE" not in llm.calls[-1]["system"] and "Python" not in llm.calls[-1]["system"]
    assert client.get("/api/users/usr_100/memories").json() == {"user_id": "usr_100", "memories": []}


def test_memories_endpoint_unknown_user_and_bad_id(client):
    r = client.get("/api/users/never_seen/memories")
    assert r.status_code == 200 and r.json() == {"user_id": "never_seen", "memories": []}
    assert client.get("/api/users/bad%20id!/memories").status_code == 422


def test_memory_outage_does_not_break_chat(client, memory_backend):
    memory_backend.fail = True
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 200
    assert r.json()["telemetry"]["memories_saved"] == 0


def test_memories_endpoint_reports_backend_failure(client, memory_backend):
    memory_backend.fail = True
    r = client.get("/api/users/usr_99/memories")
    assert r.status_code == 502 and "detail" in r.json()


def test_rerank_failure_degrades_gracefully_and_is_flagged(client, cohere_mock):
    def boom(*a, **k):
        raise RuntimeError("cohere down")
    cohere_mock.rerank = boom
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 200
    t = r.json()["telemetry"]
    assert t["rerank_degraded"] is True and t["compressed_contexts"] <= 5


def test_llm_failure_returns_502_without_secrets(client, llm):
    llm.fail = RuntimeError("401 invalid api_key sk-abcdefghijklmnop1234")
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 502
    assert "sk-abcdefghijklmnop1234" not in r.text and "detail" in r.json()


def test_missing_configuration_returns_503(client):
    def broken():
        raise ConfigurationError("Invalid configuration: COHERE_API_KEY is required")
    client.app.dependency_overrides[get_chat_service] = lambda: broken
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 503 and "COHERE_API_KEY" in r.json()["detail"]


def test_unexpected_error_returns_500_generic(client, chat_service, monkeypatch):
    monkeypatch.setattr(chat_service.agent, "answer", lambda *a, **k: (_ for _ in ()).throw(ValueError("internal sk-zzzzzzzzzzzzzz")))
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/chat" in paths and "/api/users/{user_id}/memories" in paths
    assert client.get("/docs").status_code == 200


def test_empty_retrieval_still_answers(client, chat_service, llm):
    chat_service.retrieval.retriever.hybrid.dense.docs = []
    chat_service.retrieval.retriever.hybrid.sparse.docs = []
    r = client.post("/api/chat", json=VALID)
    assert r.status_code == 200 and r.json()["sources"] == []
    assert "(no relevant passages were retrieved)" in llm.calls[-1]["user"]
