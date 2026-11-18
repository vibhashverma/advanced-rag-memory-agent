"""RAG agent: builds the persona-aware grounded prompt and generates the answer."""
from __future__ import annotations

from langchain_core.documents import Document

from app.agent.generation import Generator
from app.agent.prompt_templates import build_system_prompt, build_user_prompt
from app.memory.persona import UserPersona


class RagAgent:
    def __init__(self, generator: Generator, max_contexts: int = 5) -> None:
        self.generator = generator
        self.max_contexts = max_contexts

    def answer(self, query: str, persona: UserPersona, contexts: list[Document]) -> str:
        # Hard guarantee: never send more than max_contexts passages to the LLM.
        contexts = contexts[: self.max_contexts]
        system = build_system_prompt(persona.to_prompt())
        user = build_user_prompt(query, contexts)
        return self.generator.generate(system, user)
