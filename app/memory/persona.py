"""Dynamic persona: categorize stored memories and render them into a system-prompt block."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.memory import MemoryItem

# Ordered: first matching rule wins. Keyword rules classify free-text facts that Mem0 extracted.
_CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("formatting", re.compile(r"code[- ]only|only code|no explanation|markdown|bullet|table|\bformat|json|step[- ]by[- ]step", re.I)),
    ("response_style", re.compile(r"brief|concise|short|terse|detailed|verbose|thorough|tone|formal|casual|eli5|simple|style|response|answer", re.I)),
    ("programming_language", re.compile(r"python|java(?!script)|javascript|typescript|golang|\bgo language|rust|c\+\+|c#|kotlin|swift|ruby|php|\bsql\b|\br\b|scala|programming language|\bcode in\b", re.I)),
    ("profession", re.compile(r"work(s|ing)? (as|at|in|for)|engineer|developer|data scientist|student|manager|researcher|designer|job|occupation|role", re.I)),
    ("interest", re.compile(r"interested in|likes?|enjoys?|hobby|loves?|passionate", re.I)),
    ("location", re.compile(r"lives? in|based in|located in|from [A-Z]", re.I)),
]

_LABELS = {
    "programming_language": "Programming language",
    "response_style": "Response style",
    "formatting": "Formatting",
    "profession": "Background",
    "interest": "Interests",
    "location": "Location",
    "general": "Other",
}
_ORDER = list(_LABELS)


def categorize_fact(fact: str) -> str:
    for category, pattern in _CATEGORY_RULES:
        if pattern.search(fact):
            return category
    return "general"


@dataclass
class UserPersona:
    user_id: str
    by_category: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.by_category

    @property
    def fact_count(self) -> int:
        return sum(len(v) for v in self.by_category.values())

    def to_prompt(self) -> str:
        """Render the profile for injection into the system prompt ('' when nothing is known)."""
        if self.is_empty:
            return ""
        lines = ["USER PROFILE (from this user's long-term memory):"]
        for cat in _ORDER:
            for fact in self.by_category.get(cat, []):
                lines.append(f"- {_LABELS[cat]}: {fact}")
        lines.append(
            "Adapt every response to this profile (language, length, format) unless the user's "
            "current message explicitly asks for something different. Do not mention this profile "
            "or that you have memory of the user unless asked."
        )
        return "\n".join(lines)


def build_persona(user_id: str, memories: list[MemoryItem]) -> UserPersona:
    by_category: dict[str, list[str]] = {}
    for m in memories:
        cat = m.category if m.category in _LABELS else categorize_fact(m.fact)
        facts = by_category.setdefault(cat, [])
        if m.fact not in facts:
            facts.append(m.fact)
    return UserPersona(user_id=user_id, by_category=by_category)
