"""Circuit breaker lifecycle tests (open / half-open probe / recovery)."""

from __future__ import annotations

import time

import pytest

from app.integrations.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


def test_opens_after_threshold_failures_and_rejects_calls():
    breaker = CircuitBreaker(name="t1", failure_threshold=2, reset_timeout=60.0)

    def boom():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        breaker.call(boom)
    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state is CircuitState.OPEN
    assert breaker.failure_count == 2

    with pytest.raises(CircuitOpenError):
        breaker.call(boom)


def test_success_resets_failed_calls_below_threshold():
    breaker = CircuitBreaker(name="t2", failure_threshold=3, reset_timeout=60.0)

    def boom():
        raise RuntimeError("down")

    def ok():
        return "up"

    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.call(ok) == "up"
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_half_open_probe_success_closes_circuit():
    breaker = CircuitBreaker(name="t3", failure_threshold=1, reset_timeout=0.02)

    def boom():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        breaker.call(boom)
    assert breaker.state is CircuitState.OPEN

    time.sleep(0.04)  # reset window elapses
    assert breaker.call(lambda: "up") == "up"  # probe admitted
    assert breaker.state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens_circuit():
    breaker = CircuitBreaker(name="t4", failure_threshold=1, reset_timeout=0.02)

    def boom():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        breaker.call(boom)
    time.sleep(0.04)

    with pytest.raises(RuntimeError):
        breaker.call(boom)  # probe fails -> back to OPEN
    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.call(boom)


def test_half_open_admits_one_probe_across_processes_atomically():
    """Audit F10: two "processes" (stores sharing one Redis) and 20 threads race for the probe."""
    from concurrent.futures import ThreadPoolExecutor

    import fakeredis

    from app.integrations.circuit_breaker import CircuitBreaker, CircuitState, RedisBreakerStore

    server = fakeredis.FakeServer()
    stores = [RedisBreakerStore(fakeredis.FakeRedis(server=server)) for _ in range(2)]
    breakers = [
        CircuitBreaker("kkb-race", failure_threshold=2, reset_timeout=0.0, store=s) for s in stores
    ]
    for _ in range(2):
        breakers[0]._on_failure()
    assert breakers[1].state == CircuitState.OPEN
    with ThreadPoolExecutor(max_workers=20) as pool:
        admitted = list(pool.map(lambda i: breakers[i % 2]._allow_request(), range(20)))
    assert admitted.count(True) == 1
    assert breakers[0].state == CircuitState.HALF_OPEN
    breakers[1]._on_failure()  # the probe fails -> OPEN again, probe slot freed
    assert breakers[0].state == CircuitState.OPEN
    assert breakers[0]._allow_request() is True  # reset_timeout 0: next probe
    breakers[0]._on_success()
    assert breakers[1].state == CircuitState.CLOSED and breakers[1].failure_count == 0


def test_half_open_probe_budget_is_configurable():
    from app.integrations.circuit_breaker import CircuitBreaker, MemoryBreakerStore

    breaker = CircuitBreaker(
        "budget",
        failure_threshold=1,
        reset_timeout=0.0,
        store=MemoryBreakerStore(),
        half_open_max_calls=3,
    )
    breaker._on_failure()
    assert [breaker._allow_request() for _ in range(4)] == [True, True, True, False]
    breaker.reset()
    assert breaker.snapshot()["state"] == "CLOSED"
