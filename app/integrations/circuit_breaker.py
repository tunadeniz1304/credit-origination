"""Compact deterministic circuit breaker (preventable failure isolation).

States: CLOSED (normal) -> OPEN (failures >= threshold; rejects calls until
``reset_timeout`` elapses) -> HALF_OPEN (a single probe is admitted) ->
CLOSED on success or back to OPEN on probe failure.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from app.core.logging import get_logger


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(RuntimeError):
    """Raised when the circuit is open and the reset window has not elapsed."""


class CircuitBreaker:
    """Failure-threshold circuit breaker with a half-open probe."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = reset_timeout
        self.logger = get_logger(f"circuit.{name}")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def _transition_to_open(self) -> None:
        if self._state is not CircuitState.OPEN:
            self.logger.warning("Circuit '%s' OPEN (failures=%d)", self.name, self._failure_count)
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()

    def _allow_request(self) -> bool:
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self.logger.info("Circuit '%s' HALF_OPEN (probe allowed)", self.name)
                return True
            return False
        return True  # HALF_OPEN admits the probe

    def _on_success(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self.logger.info("Circuit '%s' closed after successful probe", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._state is CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._transition_to_open()

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke a blocking fn through the breaker."""
        if not self._allow_request():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    async def call_async(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Invoke an async fn through the breaker."""
        if not self._allow_request():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result
