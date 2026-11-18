"""Structured (key=value) logging setup."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s")
    )
    root.handlers = [handler]
    root.setLevel(level)
    root._rag_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def kv(**fields: object) -> str:
    """Render fields as key=value pairs for structured log lines."""
    return " ".join(f"{k}={v!r}" for k, v in fields.items())
