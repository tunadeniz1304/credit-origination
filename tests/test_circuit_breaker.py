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
