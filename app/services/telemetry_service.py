"""Per-request latency telemetry. Timing is measured here, never inside business logic."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from app.schemas.chat import Telemetry
from app.utils.timing import Stopwatch

STAGES = ("retrieval", "rerank", "generation", "memory_retrieval", "memory_update")


class RequestTimer:
    """Tracks total request time plus named stages using time.perf_counter().

    A stage's elapsed time is recorded even if the stage raises, so failures never look free.
    Stages that never ran stay at 0.0 and are therefore distinguishable from executed ones.
    """

    def __init__(self) -> None:
        self._total = Stopwatch().start()
        self._stages: dict[str, float] = {s: 0.0 for s in STAGES}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self._stages:
            raise KeyError(f"Unknown telemetry stage {name!r}")
        sw = Stopwatch()
        try:
            with sw:
                yield
        finally:
            self._stages[name] += sw.elapsed_ms

    def snapshot(self, **counters: int | bool) -> Telemetry:
        """Stop the total clock and build the telemetry block."""
        total = self._total.stop()  # idempotent: later calls return the already-recorded value
        r = lambda v: round(v, 3)  # noqa: E731
        return Telemetry(
            retrieval_latency_ms=r(self._stages["retrieval"]),
            rerank_latency_ms=r(self._stages["rerank"]),
            generation_latency_ms=r(self._stages["generation"]),
            total_latency_ms=r(total),
            memory_retrieval_latency_ms=r(self._stages["memory_retrieval"]),
            memory_update_latency_ms=r(self._stages["memory_update"]),
            **counters,
        )
