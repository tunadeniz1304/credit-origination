"""Hash-chained, actor-attributed audit log.

Each entry stores ``prev_hash`` and ``hash = sha256(prev_hash || canonical
entry)``. Appends are serialised per transaction: PostgreSQL takes a
transaction-scoped advisory lock; SQLite (single process) uses a process lock
held until the session commits or rolls back. :func:`verify_chain` recomputes
every link and reports the first broken entry.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from app.db.models import AuditLog, utcnow

GENESIS = "0" * 64
_ADVISORY_KEY = 4_242_2026
# A plain Lock (not RLock): FastAPI may commit a session on a different thread than
# the one that appended, and only non-owned locks can be released cross-thread.
_process_lock = threading.Lock()
_LOCK_TIMEOUT = 30.0


def _normalise_ts(ts: datetime) -> str:
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return ts.isoformat(timespec="microseconds")


def compute_hash(
    prev_hash: str,
    *,
    ts: datetime,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "ts": _normalise_ts(ts),
            "actor": actor,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def _on_transaction_end(session: Session, transaction: Any) -> None:
    """Release only when the *root* transaction ends (savepoints must not)."""
    if transaction.parent is not None:
        return
    session.info.pop("audit_pg_locked", None)
    if session.info.pop("audit_lock_held", False):
        _process_lock.release()


def _ensure_listener(session: Session) -> None:
    if not session.info.get("audit_listener"):
        event.listen(session, "after_transaction_end", _on_transaction_end)
        session.info["audit_listener"] = True


def _acquire(session: Session) -> None:
    if session.info.get("audit_lock_held") or session.info.get("audit_pg_locked"):
        return
    _ensure_listener(session)
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_KEY})
        session.info["audit_pg_locked"] = True
        return
    if not _process_lock.acquire(timeout=_LOCK_TIMEOUT):  # pragma: no cover - defensive
        return  # never hang the request; verify_chain will surface a fork
    session.info["audit_lock_held"] = True


def append_audit(
    session: Session,
    *,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: dict[str, Any] | None = None,
) -> AuditLog:
    """Append one chained entry inside the caller's transaction."""
    _acquire(session)
    payload = json.loads(json.dumps(payload or {}, ensure_ascii=False, default=str))
    last = session.execute(
        select(AuditLog.hash).order_by(AuditLog.id.desc()).limit(1)
    ).scalar_one_or_none()
    prev = last or GENESIS
    ts = utcnow()
    entry = AuditLog(
        ts=ts,
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        payload=payload,
        prev_hash=prev,
        hash=compute_hash(
            prev,
            ts=ts,
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload,
        ),
    )
    session.add(entry)
    session.flush()
    return entry


def verify_chain(session: Session) -> dict[str, Any]:
    """Recompute the whole chain; return validity and the first broken id."""
    prev = GENESIS
    count = 0
    for row in session.execute(select(AuditLog).order_by(AuditLog.id)).scalars():
        count += 1
        expected = compute_hash(
            prev,
            ts=row.ts,
            actor=row.actor,
            action=row.action,
            entity_type=row.entity_type,
            entity_id=row.entity_id,
            payload=row.payload,
        )
        if row.prev_hash != prev or row.hash != expected:
            return {"valid": False, "entries": count, "broken_at": row.id}
        prev = row.hash
    return {"valid": True, "entries": count, "broken_at": None, "head": prev}


def entity_trail(session: Session, entity_id: str) -> list[AuditLog]:
    return list(
        session.execute(
            select(AuditLog).where(AuditLog.entity_id == entity_id).order_by(AuditLog.id)
        ).scalars()
    )
