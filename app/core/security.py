"""Secret-safety helpers: make sure API keys never reach logs or API responses."""
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|token|authorization|bearer)(['\"=:\s]+)[A-Za-z0-9_\-\.]{8,}"),
]


def redact(text: str, secrets: list[str] | None = None) -> str:
    """Remove known secret values and key-looking strings from a message."""
    out = str(text)
    for s in secrets or []:
        if s and len(s) >= 6:
            out = out.replace(s, "***")
    out = _PATTERNS[0].sub("***", out)
    out = _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}***", out)
    return out
