"""Circuit breaker with pluggable, shareable, *atomic* state.

States: CLOSED (normal) -> OPEN (failures >= threshold; rejects calls until
``reset_timeout`` elapses) -> HALF_OPEN (at most ``half_open_max_calls``
probes admitted) -> CLOSED on a successful probe or back to OPEN on a failed
one. A probe that ends without an outcome (task cancelled, ``BaseException``)
gives its slot back, and every slot is a lease of ``probe_lease`` seconds, so
a probe whose process died cannot wedge the breaker in HALF_OPEN.

Every admitted call carries a *ticket*: ``C:<epoch>`` for a call admitted while
CLOSED, ``H:<lease id>`` for a half-open probe. The epoch changes whenever the
breaker opens or closes. An outcome only counts if its ticket still matches the
state it was admitted under, so a straggler admitted before the breaker opened
cannot close it (late success) or push its re-open time back (late failure),
and a probe whose lease expired cannot release or decide someone else's slot.

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
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
    epoch: int = 0  # bumped whenever the breaker opens or closes


@dataclass
class _Record:
    state: str = CircuitState.CLOSED.value
    failures: int = 0
    opened_at: float = 0.0
    epoch: int = 0
    leases: dict[str, float] = field(default_factory=dict)  # lease id -> leased at

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(
            state=self.state,
            failures=self.failures,
            opened_at=self.opened_at,
            probes=len(self.leases),
            probe_at=max(self.leases.values(), default=0.0),
            epoch=self.epoch,
        )


class BreakerStore(Protocol):
    def load(self, name: str) -> BreakerSnapshot: ...

    def acquire(
        self,
        name: str,
        now: float,
        reset_timeout: float,
        max_probes: int,
        lease: float,
        lease_id: str,
    ) -> str:
        """Atomically decide admission; returns the caller's ticket or ``""`` (rejected)."""
        ...

    def success(self, name: str, ticket: str = "") -> str:
        """Record a success; ``""`` records it unconditionally. Returns the resulting state."""
        ...

    def failure(self, name: str, now: float, threshold: int, ticket: str = "") -> str:
        """Record a failure; ``""`` records it unconditionally. Returns the resulting state."""
        ...

    def release(self, name: str, ticket: str) -> None:
        """Give back the probe slot of ``ticket`` (its call ended without an outcome)."""
        ...

    def reset(self, name: str) -> None: ...


def _ticket_matches(rec: _Record, ticket: str) -> bool:
    """Does ``ticket`` still belong to the state the breaker is in? ``""`` always matches."""
    if not ticket:
        return True
    kind, _, value = ticket.partition(":")
    if kind == "C":
        return rec.state == CircuitState.CLOSED.value and str(rec.epoch) == value
    return rec.state == CircuitState.HALF_OPEN.value and value in rec.leases


class MemoryBreakerStore:
    """Thread-safe in-process store; each transition runs under one lock."""

    def __init__(self) -> None:
        self._data: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def _get(self, name: str) -> _Record:
        return self._data.setdefault(name, _Record())

    def load(self, name: str) -> BreakerSnapshot:
        with self._lock:
            return self._get(name).snapshot()

    def acquire(
        self,
        name: str,
        now: float,
        reset_timeout: float,
        max_probes: int,
        lease: float,
        lease_id: str,
    ) -> str:
        with self._lock:
            rec = self._get(name)
            if rec.state == CircuitState.CLOSED.value:
                return f"C:{rec.epoch}"
            if rec.state == CircuitState.OPEN.value:
                if now - rec.opened_at < reset_timeout:
                    return ""
                rec.state, rec.leases = CircuitState.HALF_OPEN.value, {}
            # leases whose holder never reported back are presumed dead
            rec.leases = {k: t for k, t in rec.leases.items() if now - t < lease}
            if len(rec.leases) >= max_probes:
                return ""
            rec.leases[lease_id] = now
            return f"H:{lease_id}"

    def success(self, name: str, ticket: str = "") -> str:
        with self._lock:
            rec = self._get(name)
            if not _ticket_matches(rec, ticket):
                return rec.state  # straggler: no verdict on the current state
            if rec.state != CircuitState.CLOSED.value:
                rec.state, rec.opened_at, rec.leases = CircuitState.CLOSED.value, 0.0, {}
                rec.epoch += 1
            rec.failures = 0
            return rec.state

    def failure(self, name: str, now: float, threshold: int, ticket: str = "") -> str:
        with self._lock:
            rec = self._get(name)
            if not _ticket_matches(rec, ticket):
                return rec.state
            rec.failures += 1
            if rec.state == CircuitState.HALF_OPEN.value or rec.failures >= threshold:
                if rec.state != CircuitState.OPEN.value:
                    rec.epoch += 1
                rec.state, rec.opened_at, rec.leases = CircuitState.OPEN.value, now, {}
            return rec.state

    def release(self, name: str, ticket: str) -> None:
        with self._lock:
            rec = self._get(name)
            kind, _, lease_id = ticket.partition(":")
            if kind == "H" and rec.state == CircuitState.HALF_OPEN.value:
                rec.leases.pop(lease_id, None)

    def reset(self, name: str) -> None:
        with self._lock:
            self._data[name] = _Record()


# Shared Lua helper: does ARGV[1] (the ticket) match the current state? KEYS[1] = breaker hash,
# KEYS[2] = hash of lease id -> leased at.
_MATCHES = """
local function matches(ticket)
  if ticket == '' then return true end
  local kind, value = string.sub(ticket, 1, 1), string.sub(ticket, 3)
  local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
  if kind == 'C' then
    return state == 'CLOSED' and (redis.call('HGET', KEYS[1], 'epoch') or '0') == value
  end
  return state == 'HALF_OPEN' and redis.call('HEXISTS', KEYS[2], value) == 1
end
"""
# ARGV = now, reset_timeout, max_probes, probe lease, lease id
_ACQUIRE = """
local now = tonumber(ARGV[1])
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
if state == 'CLOSED' then return 'C:' .. (redis.call('HGET', KEYS[1], 'epoch') or '0') end
if state == 'OPEN' then
  local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
  if now - opened < tonumber(ARGV[2]) then return '' end
  redis.call('HSET', KEYS[1], 'state', 'HALF_OPEN')
  redis.call('DEL', KEYS[2])
end
local leases = redis.call('HGETALL', KEYS[2])
local live = 0
for i = 1, #leases, 2 do
  if now - tonumber(leases[i + 1]) >= tonumber(ARGV[4]) then
    redis.call('HDEL', KEYS[2], leases[i])
  else
    live = live + 1
  end
end
if live >= tonumber(ARGV[3]) then return '' end
redis.call('HSET', KEYS[2], ARGV[5], ARGV[1])
return 'H:' .. ARGV[5]
"""
# ARGV = ticket
_RELEASE = """
local kind, value = string.sub(ARGV[1], 1, 1), string.sub(ARGV[1], 3)
if kind == 'H' and redis.call('HGET', KEYS[1], 'state') == 'HALF_OPEN' then
  redis.call('HDEL', KEYS[2], value)
end
return 1
"""
# ARGV = ticket
_SUCCESS = (
    _MATCHES
    + """
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
if not matches(ARGV[1]) then return state end
if state ~= 'CLOSED' then
  redis.call('HSET', KEYS[1], 'state', 'CLOSED', 'opened_at', 0)
  redis.call('HINCRBY', KEYS[1], 'epoch', 1)
  redis.call('DEL', KEYS[2])
end
redis.call('HSET', KEYS[1], 'failures', 0)
return 'CLOSED'
"""
)
# ARGV = ticket, now, threshold
_FAILURE = (
    _MATCHES
    + """
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
if not matches(ARGV[1]) then return state end
local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
if state == 'HALF_OPEN' or failures >= tonumber(ARGV[3]) then
  if state ~= 'OPEN' then redis.call('HINCRBY', KEYS[1], 'epoch', 1) end
  redis.call('HSET', KEYS[1], 'state', 'OPEN', 'opened_at', ARGV[2])
  redis.call('DEL', KEYS[2])
  return 'OPEN'
end
if state == 'CLOSED' then redis.call('HSET', KEYS[1], 'state', 'CLOSED') end
return state
"""
)


class RedisBreakerStore:
    """Cross-process store: one hash per breaker (+ one for leases), transitions as Lua scripts."""

    def __init__(self, client: Any, prefix: str = "anil2:circuit:") -> None:
        self._client = client
        self._prefix = prefix
        self._acquire = client.register_script(_ACQUIRE)
        self._success = client.register_script(_SUCCESS)
        self._failure = client.register_script(_FAILURE)
        self._release = client.register_script(_RELEASE)

    def _keys(self, name: str) -> list[str]:
        return [self._prefix + name, self._prefix + name + ":leases"]

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value or "")

    def load(self, name: str) -> BreakerSnapshot:
        key, lease_key = self._keys(name)
        raw = {self._text(k): self._text(v) for k, v in self._client.hgetall(key).items()}
        leases = [float(self._text(v)) for v in self._client.hgetall(lease_key).values()]
        if not raw:
            return BreakerSnapshot()
        return BreakerSnapshot(
            state=raw.get("state", CircuitState.CLOSED.value),
            failures=int(raw.get("failures", 0) or 0),
            opened_at=float(raw.get("opened_at", 0) or 0),
            probes=len(leases),
            probe_at=max(leases, default=0.0),
            epoch=int(raw.get("epoch", 0) or 0),
        )

    def acquire(
        self,
        name: str,
        now: float,
        reset_timeout: float,
        max_probes: int,
        lease: float,
        lease_id: str,
    ) -> str:
        return self._text(
            self._acquire(
                keys=self._keys(name), args=[now, reset_timeout, max_probes, lease, lease_id]
            )
        )

    def success(self, name: str, ticket: str = "") -> str:
        return self._text(self._success(keys=self._keys(name), args=[ticket]))

    def failure(self, name: str, now: float, threshold: int, ticket: str = "") -> str:
        return self._text(self._failure(keys=self._keys(name), args=[ticket, now, threshold]))

    def release(self, name: str, ticket: str) -> None:
        self._release(keys=self._keys(name), args=[ticket])

    def reset(self, name: str) -> None:
        self._client.delete(*self._keys(name))


_FailurePredicate = Callable[[Exception], bool]


def _always(_: Exception) -> bool:
    return True


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

    def _admit(self) -> str:
        """Ask the store for admission; returns the call's ticket or ``""``."""
        ticket = self._store.acquire(
            self.name,
            time.time(),
            self.reset_timeout,
            self.half_open_max_calls,
            self.probe_lease,
            uuid.uuid4().hex,
        )
        if ticket.startswith("H:"):
            self._publish(CircuitState.HALF_OPEN.value)
            self.logger.info("Circuit '%s' HALF_OPEN (probe admitted)", self.name)
        return ticket

    def _allow_request(self) -> bool:
        return bool(self._admit())

    def _on_success(self, ticket: str = "") -> None:
        self._publish(self._store.success(self.name, ticket))

    def _on_failure(self, ticket: str = "") -> None:
        state = self._store.failure(self.name, time.time(), self.failure_threshold, ticket)
        if state == CircuitState.OPEN.value:
            self.logger.warning("Circuit '%s' OPEN", self.name)
        self._publish(state)

    def _settle(
        self, ticket: str, error: BaseException | None, is_failure: _FailurePredicate
    ) -> None:
        """Record the outcome of an admitted call.

        Only an ``Exception`` that ``is_failure`` accepts counts against the service; any other
        ending (cancellation, or e.g. an HTTP 4xx the caller caused) gives the slot back.
        """
        if error is None:
            self._on_success(ticket)
        elif isinstance(error, Exception) and is_failure(error):
            self._on_failure(ticket)
        else:
            self._store.release(self.name, ticket)

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        is_failure: _FailurePredicate = _always,
        **kwargs: Any,
    ) -> Any:
        """Invoke a blocking fn through the breaker."""
        ticket = self._admit()
        if not ticket:
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            self._settle(ticket, exc, is_failure)
            raise
        self._settle(ticket, None, is_failure)
        return result

    async def call_async(
        self,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        is_failure: _FailurePredicate = _always,
        **kwargs: Any,
    ) -> Any:
        """Invoke an async fn through the breaker."""
        ticket = self._admit()
        if not ticket:
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:
            self._settle(ticket, exc, is_failure)
            raise
        self._settle(ticket, None, is_failure)
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
