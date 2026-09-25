"""Circuit breaker with pluggable, shareable, *atomic* state.

States: CLOSED (normal) -> OPEN (failures >= threshold; rejects calls until
``reset_timeout`` elapses) -> HALF_OPEN (at most ``half_open_max_calls``
probes admitted) -> CLOSED on a successful probe or back to OPEN on a failed
one. A probe that ends without an outcome (task cancelled, ``BaseException``)
gives its slot back, and every slot is a lease of ``probe_lease`` seconds, so
a probe whose process died cannot wedge the breaker in HALF_OPEN.

Every transition is a single atomic store operation. v1 did load → modify →
save from Python, so two processes could both flip OPEN → HALF_OPEN and
HALF_OPEN admitted every caller. :class:`RedisBreakerStore` now runs each
transition as a Lua script (executed atomically by Redis), and
:class:`MemoryBreakerStore` performs the same logic under one lock.

:func:`get_breaker` returns one breaker per service name for the whole process
(so failures accumulate across requests); with ``CIRCUIT_STATE_BACKEND=redis``
(or ``auto`` with a reachable Redis, as in the Celery topology) the state is
shared across processes.
"""

from __future__ import annotations

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
    """Raised when the circuit is open (or its probe slots are taken)."""


@dataclass
class BreakerSnapshot:
    state: str = CircuitState.CLOSED.value
    failures: int = 0
    opened_at: float = 0.0
    probes: int = 0  # probes in flight while HALF_OPEN
    probe_at: float = 0.0  # when the latest probe slot was leased


class BreakerStore(Protocol):
    def load(self, name: str) -> BreakerSnapshot: ...

    def acquire(
        self, name: str, now: float, reset_timeout: float, max_probes: int, lease: float
    ) -> str:
        """Atomically decide admission; returns the state the caller runs under or ``""``."""
        ...

    def success(self, name: str) -> str: ...

    def failure(self, name: str, now: float, threshold: int) -> str: ...

    def release(self, name: str) -> None:
        """Give back a probe slot whose call ended without an outcome."""
        ...

    def reset(self, name: str) -> None: ...


def _decide(
    snap: BreakerSnapshot, now: float, reset_timeout: float, max_probes: int, lease: float
) -> tuple[BreakerSnapshot, str]:
    if snap.state == CircuitState.CLOSED.value:
        return snap, CircuitState.CLOSED.value
    if snap.state == CircuitState.OPEN.value:
        if now - snap.opened_at < reset_timeout:
            return snap, ""
        snap.state, snap.probes = CircuitState.HALF_OPEN.value, 0
    if snap.probes >= max_probes:
        if now - snap.probe_at < lease:
            return snap, ""
        snap.probes = 0  # the leases expired: their probes are presumed dead
    snap.probes += 1
    snap.probe_at = now
    return snap, CircuitState.HALF_OPEN.value


def _on_failure(snap: BreakerSnapshot, now: float, threshold: int) -> BreakerSnapshot:
    snap.failures += 1
    if snap.state == CircuitState.HALF_OPEN.value or snap.failures >= threshold:
        snap.state, snap.opened_at, snap.probes = CircuitState.OPEN.value, now, 0
    return snap


class MemoryBreakerStore:
    """Thread-safe in-process store; each transition runs under one lock."""

    def __init__(self) -> None:
        self._data: dict[str, BreakerSnapshot] = {}
        self._lock = threading.Lock()

    def _get(self, name: str) -> BreakerSnapshot:
        return self._data.setdefault(name, BreakerSnapshot())

    def load(self, name: str) -> BreakerSnapshot:
        with self._lock:
            return BreakerSnapshot(**asdict(self._get(name)))

    def acquire(
        self, name: str, now: float, reset_timeout: float, max_probes: int, lease: float
    ) -> str:
        with self._lock:
            snap, admitted = _decide(self._get(name), now, reset_timeout, max_probes, lease)
            self._data[name] = snap
            return admitted

    def success(self, name: str) -> str:
        with self._lock:
            self._data[name] = BreakerSnapshot()
            return CircuitState.CLOSED.value

    def failure(self, name: str, now: float, threshold: int) -> str:
        with self._lock:
            snap = _on_failure(self._get(name), now, threshold)
            self._data[name] = snap
            return snap.state

    def release(self, name: str) -> None:
        with self._lock:
            snap = self._get(name)
            if snap.state == CircuitState.HALF_OPEN.value and snap.probes > 0:
                snap.probes -= 1

    def reset(self, name: str) -> None:
        with self._lock:
            self._data[name] = BreakerSnapshot()


# KEYS[1] = breaker hash; ARGV = now, reset_timeout, max_probes, probe lease
_ACQUIRE = """
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
if state == 'CLOSED' then return 'CLOSED' end
if state == 'OPEN' then
  local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
  if tonumber(ARGV[1]) - opened < tonumber(ARGV[2]) then return '' end
  redis.call('HSET', KEYS[1], 'state', 'HALF_OPEN', 'probes', 0)
end
local probes = tonumber(redis.call('HGET', KEYS[1], 'probes') or '0')
if probes >= tonumber(ARGV[3]) then
  local leased = tonumber(redis.call('HGET', KEYS[1], 'probe_at') or '0')
  if tonumber(ARGV[1]) - leased < tonumber(ARGV[4]) then return '' end
  redis.call('HSET', KEYS[1], 'probes', 0)
end
redis.call('HINCRBY', KEYS[1], 'probes', 1)
redis.call('HSET', KEYS[1], 'probe_at', ARGV[1])
return 'HALF_OPEN'
"""
_RELEASE = """
if redis.call('HGET', KEYS[1], 'state') == 'HALF_OPEN'
   and tonumber(redis.call('HGET', KEYS[1], 'probes') or '0') > 0 then
  redis.call('HINCRBY', KEYS[1], 'probes', -1)
end
return 1
"""
_SUCCESS = """
redis.call('HSET', KEYS[1], 'state', 'CLOSED', 'failures', 0, 'opened_at', 0, 'probes', 0)
return 'CLOSED'
"""
# ARGV = now, threshold
_FAILURE = """
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
if state == 'HALF_OPEN' or failures >= tonumber(ARGV[2]) then
  redis.call('HSET', KEYS[1], 'state', 'OPEN', 'opened_at', ARGV[1], 'probes', 0)
  return 'OPEN'
end
if state == 'CLOSED' then redis.call('HSET', KEYS[1], 'state', 'CLOSED') end
return state
"""


class RedisBreakerStore:
    """Cross-process store: one hash per breaker, transitions as Lua scripts."""

    def __init__(self, client: Any, prefix: str = "anil2:circuit:") -> None:
        self._client = client
        self._prefix = prefix
        self._acquire = client.register_script(_ACQUIRE)
        self._success = client.register_script(_SUCCESS)
        self._failure = client.register_script(_FAILURE)
        self._release = client.register_script(_RELEASE)

    def _key(self, name: str) -> str:
        return self._prefix + name

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value or "")

    def load(self, name: str) -> BreakerSnapshot:
        raw = {
            self._text(k): self._text(v) for k, v in self._client.hgetall(self._key(name)).items()
        }
        if not raw:
            return BreakerSnapshot()
        return BreakerSnapshot(
            state=raw.get("state", CircuitState.CLOSED.value),
            failures=int(raw.get("failures", 0) or 0),
            opened_at=float(raw.get("opened_at", 0) or 0),
            probes=int(raw.get("probes", 0) or 0),
            probe_at=float(raw.get("probe_at", 0) or 0),
        )

    def acquire(
        self, name: str, now: float, reset_timeout: float, max_probes: int, lease: float
    ) -> str:
        return self._text(
            self._acquire(keys=[self._key(name)], args=[now, reset_timeout, max_probes, lease])
        )

    def success(self, name: str) -> str:
        return self._text(self._success(keys=[self._key(name)]))

    def failure(self, name: str, now: float, threshold: int) -> str:
        return self._text(self._failure(keys=[self._key(name)], args=[now, threshold]))

    def release(self, name: str) -> None:
        self._release(keys=[self._key(name)])

    def reset(self, name: str) -> None:
        self._client.delete(self._key(name))


class CircuitBreaker:
    """Failure-threshold circuit breaker with a bounded half-open probe."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        reset_timeout: float = 30.0,
        store: BreakerStore | None = None,
        half_open_max_calls: int = 1,
        probe_lease: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = max(1, half_open_max_calls)
        self.probe_lease = probe_lease
        self.logger = get_logger(f"circuit.{name}")
        self._store: BreakerStore = store or MemoryBreakerStore()

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
        admitted = self._store.acquire(
            self.name, time.time(), self.reset_timeout, self.half_open_max_calls, self.probe_lease
        )
        if admitted == CircuitState.HALF_OPEN.value:
            self._publish(admitted)
            self.logger.info("Circuit '%s' HALF_OPEN (probe admitted)", self.name)
        return bool(admitted)

    def _on_success(self) -> None:
        self._publish(self._store.success(self.name))

    def _on_failure(self) -> None:
        state = self._store.failure(self.name, time.time(), self.failure_threshold)
        if state == CircuitState.OPEN.value:
            self.logger.warning("Circuit '%s' OPEN", self.name)
        self._publish(state)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke a blocking fn through the breaker."""
        if not self._allow_request():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        except BaseException:  # cancelled / interrupted: no verdict on the service
            self._store.release(self.name)
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
        except BaseException:  # cancelled / interrupted: no verdict on the service
            self._store.release(self.name)
            raise
        self._on_success()
        return result

    def reset(self) -> None:
        self._store.reset(self.name)


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
    from app.core.config import get_settings

    with _registry_lock:
        breaker = _registry.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name,
                failure_threshold,
                reset_timeout,
                store=_resolve_store(),
                half_open_max_calls=get_settings().circuit_half_open_max_calls,
            )
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
