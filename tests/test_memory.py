"""TEST 4: semantic memory (Mem0 manager + persona)."""
import pytest

from app.memory.mem0_manager import Mem0Manager, MemoryError_, MemoryNotFound
from app.memory.persona import build_persona, categorize_fact
from app.schemas.memory import MemoryItem

MSG = "Hello! I work in Python and prefer brief, code-only responses."


def facts(items):
    return {m.fact for m in items}


def test_facts_extracted_saved_and_categorized(memory_manager):
    saved = memory_manager.add_memory("usr_99", MSG, session_id="session_001")
    assert facts(saved) == {"Prefers Python", "Prefers brief", "Prefers code-only responses"}
    cats = {m.fact: m.category for m in memory_manager.get_all_memories("usr_99")}
    assert cats["Prefers Python"] == "programming_language"
    assert cats["Prefers code-only responses"] == "formatting"


def test_memories_persist_across_sessions(memory_manager, memory_backend):
    memory_manager.add_memory("usr_99", MSG, session_id="session_001")
    # A brand-new session: retrieval takes only user_id, so session_001's facts are visible.
    assert "Prefers Python" in facts(memory_manager.get_all_memories("usr_99"))
    # ...and session id is stored as metadata, not used as a filter.
    assert {r["metadata"]["session_id"] for r in memory_backend.rows.values()} == {"session_001"}
    memory_manager.add_memory("usr_99", "I prefer tabs", session_id="session_002")
    assert "Prefers tabs" in facts(memory_manager.get_all_memories("usr_99"))
    assert "Prefers Python" in facts(memory_manager.get_all_memories("usr_99"))


def test_users_are_isolated(memory_manager):
    memory_manager.add_memory("alice", "I work in Python", session_id="s1")
    memory_manager.add_memory("bob", "I work in Rust", session_id="s1")
    assert facts(memory_manager.get_all_memories("alice")) == {"Prefers Python"}
    assert facts(memory_manager.get_all_memories("bob")) == {"Prefers Rust"}
    assert facts(memory_manager.search_memory("alice", "Rust")) == set()


def test_manager_filters_foreign_records_even_if_backend_leaks(memory_backend):
    class Leaky:
        def get_all(self, user_id, limit):
            return [{"id": "1", "memory": "Prefers Python", "user_id": user_id},
                    {"id": "2", "memory": "Secret of bob", "user_id": "bob"}]
    assert facts(Mem0Manager(Leaky()).get_all_memories("alice")) == {"Prefers Python"}


def test_empty_results_handled(memory_manager):
    assert memory_manager.get_all_memories("nobody") == []
    assert memory_manager.search_memory("nobody", "python") == []
    assert memory_manager.add_memory("u", "   ") == []
    assert memory_manager.add_memory("u", "What is the weather today?") == []  # nothing durable
    assert memory_manager.get_all_memories("u") == []


def test_search_memory(memory_manager):
    memory_manager.add_memory("u", MSG)
    assert "Prefers Python" in facts(memory_manager.search_memory("u", "which Python version", limit=5))


def test_update_and_delete_require_ownership(memory_manager):
    memory_manager.add_memory("alice", "I work in Python")
    mem_id = memory_manager.get_all_memories("alice")[0].id
    with pytest.raises(MemoryNotFound):
        memory_manager.update_memory("bob", mem_id, "hacked")
    with pytest.raises(MemoryNotFound):
        memory_manager.delete_memory("bob", mem_id)
    memory_manager.update_memory("alice", mem_id, "Prefers Go")
    assert facts(memory_manager.get_all_memories("alice")) == {"Prefers Go"}
    memory_manager.delete_memory("alice", mem_id)
    assert memory_manager.get_all_memories("alice") == []


def test_blank_user_id_rejected(memory_manager):
    with pytest.raises(ValueError):
        memory_manager.get_all_memories("  ")


def test_backend_failure_wrapped(memory_manager, memory_backend):
    memory_backend.fail = True
    with pytest.raises(MemoryError_):
        memory_manager.get_all_memories("u")


def test_persona_is_dynamic_not_hardcoded():
    p = build_persona("u1", [MemoryItem(fact="Prefers Java", category="programming_language"),
                             MemoryItem(fact="Prefers detailed explanations", category="response_style")])
    text = p.to_prompt()
    assert "Java" in text and "detailed explanations" in text
    assert "Python" not in text and "code-only" not in text.lower()
    assert build_persona("u2", []).to_prompt() == "" and build_persona("u2", []).is_empty


@pytest.mark.parametrize("fact,cat", [
    ("Prefers Python", "programming_language"),
    ("Prefers brief responses", "response_style"),
    ("Prefers code-only responses", "formatting"),
    ("Is a data scientist", "profession"),
    ("Has a cat named Tom", "general"),
])
def test_categorize_fact(fact, cat):
    assert categorize_fact(fact) == cat
