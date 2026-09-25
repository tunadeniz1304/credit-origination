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


def _stores():
    import fakeredis

    from app.integrations.circuit_breaker import MemoryBreakerStore, RedisBreakerStore

    return [MemoryBreakerStore(), RedisBreakerStore(fakeredis.FakeRedis())]


@pytest.mark.parametrize("store_index", [0, 1])
def test_cancelled_probe_releases_its_slot(store_index):
    """Audit v2.1 round 1: a cancelled probe left HALF_OPEN with probes=1 forever."""
    import asyncio

    store = _stores()[store_index]
    breaker = CircuitBreaker("cancel", failure_threshold=1, reset_timeout=0.0, store=store)
    breaker._on_failure()

    async def hang():
        await asyncio.sleep(10)

    async def run():
        task = asyncio.create_task(breaker.call_async(hang))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert breaker.state == CircuitState.HALF_OPEN
    assert store.load("cancel").probes == 0
    assert breaker.call(lambda: "ok") == "ok"  # the next caller can probe
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.parametrize("store_index", [0, 1])
def test_probe_lease_expires_when_its_process_died(store_index):
    store = _stores()[store_index]
    breaker = CircuitBreaker(
        "lease", failure_threshold=1, reset_timeout=0.0, store=store, probe_lease=0.05
    )
    breaker._on_failure()
    assert breaker._allow_request() is True  # probe admitted, then its process "dies"
    assert breaker._allow_request() is False  # lease still held
    time.sleep(0.08)
    assert breaker._allow_request() is True  # lease expired: a new probe may run


@pytest.mark.parametrize("store_index", [0, 1])
def test_straggler_outcomes_do_not_decide_a_newer_state(store_index):
    """Audit v2.1 round 2: a call admitted while CLOSED finished after the breaker opened."""
    store = _stores()[store_index]
    breaker = CircuitBreaker("late", failure_threshold=1, reset_timeout=60.0, store=store)
    early_ok, early_bad = breaker._admit(), breaker._admit()
    assert early_ok.startswith("C:") and early_bad.startswith("C:")
    breaker._on_failure(breaker._admit())  # a third call fails -> OPEN
    opened_at = store.load("late").opened_at
    assert breaker.state == CircuitState.OPEN

    time.sleep(0.01)
    breaker._on_failure(early_bad)  # late failure must not push the re-open time back
    assert store.load("late").opened_at == opened_at
    breaker._on_success(early_ok)  # late success must not close the open breaker
    assert breaker.state == CircuitState.OPEN


@pytest.mark.parametrize("store_index", [0, 1])
def test_stale_probe_cannot_release_or_decide_the_new_probe(store_index):
    store = _stores()[store_index]
    breaker = CircuitBreaker(
        "stale", failure_threshold=1, reset_timeout=0.0, store=store, probe_lease=0.05
    )
    breaker._on_failure()
    stale = breaker._admit()
    time.sleep(0.08)  # the stale probe's lease expires
    fresh = breaker._admit()
    assert stale.startswith("H:") and fresh.startswith("H:") and stale != fresh

    store.release("stale", stale)  # must not free the fresh probe's slot
    assert store.load("stale").probes == 1
    assert breaker._allow_request() is False
    breaker._on_success(stale)  # nor may its late verdict close the breaker
    assert breaker.state == CircuitState.HALF_OPEN
    breaker._on_success(fresh)
    assert breaker.state == CircuitState.CLOSED


def test_client_errors_do_not_open_the_breaker():
    """Audit v2.1 round 2: every HTTPStatusError, 4xx included, counted as an outage."""
    import asyncio

    import httpx

    from app.integrations.clients import _counts_against_service

    breaker = CircuitBreaker("http", failure_threshold=1, reset_timeout=60.0)
    request = httpx.Request("GET", "http://svc/x")

    def status_error(code: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code))

    async def fails_with(code: int):
        raise status_error(code)

    for code in (400, 404, 422):
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(breaker.call_async(fails_with, code, is_failure=_counts_against_service))
    assert breaker.state == CircuitState.CLOSED and breaker.failure_count == 0
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(breaker.call_async(fails_with, 503, is_failure=_counts_against_service))
    assert breaker.state == CircuitState.OPEN
    assert _counts_against_service(httpx.ConnectError("down")) is True
