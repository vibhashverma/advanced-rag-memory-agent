"""Timing utilities, kept separate from business logic."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Stopwatch:
    """Measures elapsed wall time with time.perf_counter(). Works as a context manager."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed_ms: float = 0.0

    def start(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            self._start = None
        return self.elapsed_ms

    def __enter__(self) -> "Stopwatch":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        # Elapsed time is recorded even when the block raised; exception propagates.
        self.stop()


@contextmanager
def timed() -> Iterator[Stopwatch]:
    sw = Stopwatch()
    with sw:
        yield sw
