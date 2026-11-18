"""LLM construction (provider chosen by env) and safe generation."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import ConfigurationError, Settings
from app.core.logging import get_logger, kv
from app.core.security import redact

log = get_logger(__name__)


class GenerationError(RuntimeError):
    """LLM call failed; message is redacted and safe to return to clients."""


def build_chat_model(settings: Settings) -> BaseChatModel:
    p = settings.llm_provider
    if p == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": settings.llm_model, "api_key": settings.openai_api_key, "temperature": 0.2}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.llm_model, api_key=settings.anthropic_api_key, temperature=0.2)
    if p == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(model=settings.llm_model, cohere_api_key=settings.cohere_api_key, temperature=0.2)
    raise ConfigurationError(f"Unsupported LLM_PROVIDER {p!r}")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks (e.g. Anthropic)
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class Generator:
    def __init__(self, llm: BaseChatModel, secrets: list[str] | None = None) -> None:
        self.llm = llm
        self._secrets = secrets or []

    def generate(self, system: str, user: str) -> str:
        try:
            msg = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as exc:
            safe = redact(f"{type(exc).__name__}: {exc}", self._secrets)
            log.error(kv(event="generation_failed", error=safe))
            raise GenerationError(f"The language model request failed ({type(exc).__name__}).") from None
        answer = _text(msg.content).strip()
        if not answer:
            raise GenerationError("The language model returned an empty response.")
        return answer
