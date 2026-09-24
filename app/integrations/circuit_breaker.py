"""Circuit breaker with pluggable, shareable state.

States: CLOSED (normal) -> OPEN (failures >= threshold; rejects calls until
``reset_timeout`` elapses) -> HALF_OPEN (one probe admitted) -> CLOSED on
success or back to OPEN on probe failure.

State lives in a :class:`BreakerStore`. :func:`get_breaker` returns one
breaker per service name for the whole process (so failures accumulate across
requests); with ``CIRCUIT_STATE_BACKEND=redis`` (or ``auto`` with a reachable
Redis, as in the Celery topology) the state is shared across processes.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

from app.core.logging import get_logger


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(RuntimeError):
    """Raised when the circuit is open and the reset window has not elapsed."""


@dataclass
class BreakerSnapshot:
    state: str = CircuitState.CLOSED.value
    failures: int = 0
    opened_at: float = 0.0


class BreakerStore(Protocol):
    def load(self, name: str) -> BreakerSnapshot: ...

    def save(self, name: str, snapshot: BreakerSnapshot) -> None: ...


class MemoryBreakerStore:
    """Thread-safe in-process store."""

    def __init__(self) -> None:
        self._data: dict[str, BreakerSnapshot] = {}
        self._lock = threading.Lock()

    def load(self, name: str) -> BreakerSnapshot:
        with self._lock:
            snap = self._data.get(name)
            return BreakerSnapshot(**asdict(snap)) if snap else BreakerSnapshot()

    def save(self, name: str, snapshot: BreakerSnapshot) -> None:
        with self._lock:
            self._data[name] = BreakerSnapshot(**asdict(snapshot))


class RedisBreakerStore:
    """Cross-process store (one JSON value per breaker)."""

    def __init__(self, client: Any, prefix: str = "anil2:circuit:") -> None:
        self._client = client
        self._prefix = prefix

    def load(self, name: str) -> BreakerSnapshot:
        raw = self._client.get(self._prefix + name)
        if not raw:
            return BreakerSnapshot()
        return BreakerSnapshot(**json.loads(raw))

    def save(self, name: str, snapshot: BreakerSnapshot) -> None:
        self._client.set(self._prefix + name, json.dumps(asdict(snapshot)))


class CircuitBreaker:
    """Failure-threshold circuit breaker with a half-open probe."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
        store: BreakerStore | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = reset_timeout
        self.logger = get_logger(f"circuit.{name}")
        self._store: BreakerStore = store or MemoryBreakerStore()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        return CircuitState(self._store.load(self.name).state)

    @property
    def failure_count(self) -> int:
        return self._store.load(self.name).failures

    def snapshot(self) -> dict[str, Any]:
        snap = self._store.load(self.name)
        return {"service": self.name, "state": snap.state, "failures": snap.failures}

    def _publish(self, state: str) -> None:
        try:
            from app.core.metrics import CIRCUIT_STATE

            CIRCUIT_STATE.labels(service=self.name).set(
                {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}[state]
            )
        except Exception:  # metrics are best effort
            self.logger.debug("circuit metric update failed", exc_info=True)

    def _allow_request(self) -> bool:
        with self._lock:
            snap = self._store.load(self.name)
            if snap.state == CircuitState.CLOSED.value:
                return True
            if snap.state == CircuitState.OPEN.value:
                if time.time() - snap.opened_at >= self.reset_timeout:
                    snap.state = CircuitState.HALF_OPEN.value
                    self._store.save(self.name, snap)
                    self._publish(snap.state)
                    self.logger.info("Circuit '%s' HALF_OPEN (probe allowed)", self.name)
                    return True
                return False
            return True

    def _on_success(self) -> None:
        with self._lock:
            snap = self._store.load(self.name)
            if snap.state == CircuitState.HALF_OPEN.value:
                self.logger.info("Circuit '%s' closed after successful probe", self.name)
            self._store.save(self.name, BreakerSnapshot())
            self._publish(CircuitState.CLOSED.value)

    def _on_failure(self) -> None:
        with self._lock:
            snap = self._store.load(self.name)
            snap.failures += 1
            if (
                snap.state == CircuitState.HALF_OPEN.value
                or snap.failures >= self.failure_threshold
            ):
                if snap.state != CircuitState.OPEN.value:
                    self.logger.warning("Circuit '%s' OPEN (failures=%d)", self.name, snap.failures)
                snap.state = CircuitState.OPEN.value
                snap.opened_at = time.time()
            self._store.save(self.name, snap)
            self._publish(snap.state)

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

    def reset(self) -> None:
        self._store.save(self.name, BreakerSnapshot())


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()
_shared_store: BreakerStore | None = None


def _resolve_store() -> BreakerStore:
    global _shared_store
    if _shared_store is not None:
        return _shared_store
    from app.core.config import get_settings

    settings = get_settings()
    backend = settings.circuit_state_backend
    store: BreakerStore = MemoryBreakerStore()
    if backend in ("redis", "auto"):
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
            client.ping()
            store = RedisBreakerStore(client)
        except Exception:
            if backend == "redis":
                raise
    _shared_store = store
    return store


def get_breaker(
    name: str, failure_threshold: int = 3, reset_timeout: float = 30.0
) -> CircuitBreaker:
    """Process-wide breaker per service name (shared state across requests)."""
    with _registry_lock:
        breaker = _registry.get(name)
        if breaker is None:
            breaker = CircuitBreaker(name, failure_threshold, reset_timeout, store=_resolve_store())
            _registry[name] = breaker
        return breaker


def all_breakers() -> list[dict[str, Any]]:
    return [b.snapshot() for b in _registry.values()]


def reset_registry(store: BreakerStore | None = None) -> None:
    """Testing hook: drop cached breakers (optionally inject a store)."""
    global _shared_store
    with _registry_lock:
        _registry.clear()
        _shared_store = store
