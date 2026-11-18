"""TEST 6: latency telemetry uses real measurements."""
import time

import pytest

from app.services.telemetry_service import RequestTimer
from app.utils.timing import Stopwatch

REQUIRED = ("retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms", "total_latency_ms")


def test_stopwatch_measures_real_elapsed_time():
    with Stopwatch() as sw:
        time.sleep(0.02)
    assert 15 <= sw.elapsed_ms < 500


def test_all_four_metrics_present_numeric_nonnegative():
    t = RequestTimer()
    for s in ("retrieval", "rerank", "generation"):
        with t.stage(s):
            time.sleep(0.003)
    snap = t.snapshot().model_dump()
    for k in REQUIRED:
        assert isinstance(snap[k], float) and snap[k] >= 0


def test_stage_values_reflect_actual_durations_not_constants():
    t = RequestTimer()
    with t.stage("retrieval"):
        time.sleep(0.03)
    with t.stage("generation"):
        time.sleep(0.01)
    snap = t.snapshot()
    assert snap.retrieval_latency_ms >= 25 and snap.generation_latency_ms >= 8
    assert snap.retrieval_latency_ms > snap.generation_latency_ms
    assert snap.rerank_latency_ms == 0.0  # never executed -> not reported as work


def test_total_covers_whole_request_including_untimed_gaps():
    t = RequestTimer()
    with t.stage("retrieval"):
        time.sleep(0.01)
    time.sleep(0.03)  # work outside any stage still counts toward total
    with t.stage("generation"):
        time.sleep(0.01)
    snap = t.snapshot()
    stages = snap.retrieval_latency_ms + snap.rerank_latency_ms + snap.generation_latency_ms
    assert snap.total_latency_ms >= stages + 25
    assert snap.total_latency_ms >= 45


def test_failed_stage_still_records_time_and_propagates():
    t = RequestTimer()
    with pytest.raises(RuntimeError):
        with t.stage("generation"):
            time.sleep(0.01)
            raise RuntimeError("boom")
    assert t.snapshot().generation_latency_ms >= 8


def test_unknown_stage_rejected():
    with pytest.raises(KeyError):
        with RequestTimer().stage("nope"):
            pass


def test_snapshot_is_idempotent():
    t = RequestTimer()
    time.sleep(0.005)
    assert t.snapshot().total_latency_ms == t.snapshot().total_latency_ms


def test_api_response_has_measured_positive_stage_latencies(client):
    body = client.post("/api/chat", json={"query": "q", "user_id": "u", "session_id": "s"}).json()
    t = body["telemetry"]
    for k in REQUIRED:
        assert isinstance(t[k], (int, float)) and t[k] > 0
    # fake dense/sparse sleep 2ms each, fake LLM sleeps 2ms -> real measurements reflect it
    assert t["retrieval_latency_ms"] >= 3.5 and t["generation_latency_ms"] >= 1.5
    assert t["total_latency_ms"] >= t["retrieval_latency_ms"] + t["rerank_latency_ms"] + t["generation_latency_ms"]


def test_two_requests_give_different_measurements(client):
    p = {"query": "q", "user_id": "u", "session_id": "s"}
    a = client.post("/api/chat", json=p).json()["telemetry"]["total_latency_ms"]
    b = client.post("/api/chat", json=p).json()["telemetry"]["total_latency_ms"]
    assert a != b
