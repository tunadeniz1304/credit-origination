"""Hash-chained, actor-attributed audit log.

Each entry stores ``prev_hash`` and ``hash = sha256(prev_hash || canonical
entry)``. Appends are serialised per transaction: PostgreSQL takes a
transaction-scoped advisory lock; SQLite relies on the process-wide writer
lock of :mod:`app.db.session` (taken before the first write of the unit of
work, so the lock order is always the same). The insert runs inside a
SAVEPOINT and is retried with jittered back-off when the database reports a
lock (e.g. another process writing). :func:`verify_chain` recomputes every
link and reports the first broken entry.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import event, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import AuditLog, utcnow
from app.db.session import _acquire_writer

GENESIS = "0" * 64
_ADVISORY_KEY = 4_242_2026


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


def _acquire(session: Session) -> None:
    if session.info.get("audit_pg_locked"):
        return
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_KEY})
        session.info["audit_pg_locked"] = True
        event.listen(session, "after_transaction_end", _forget_pg_lock, once=True)
        return
    _acquire_writer(session)


def _forget_pg_lock(session: Session, transaction: Any) -> None:
    session.info.pop("audit_pg_locked", None)


def _is_lock_error(exc: OperationalError) -> bool:
    return "locked" in str(exc).lower() or "busy" in str(exc).lower()


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
    settings = get_settings()
    _acquire(session)
    payload = json.loads(json.dumps(payload or {}, ensure_ascii=False, default=str))
    for attempt in range(settings.audit_lock_retries + 1):
        try:
            with session.begin_nested():
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
        except OperationalError as exc:
            if not _is_lock_error(exc) or attempt >= settings.audit_lock_retries:
                raise
            delay = settings.audit_retry_base_seconds * (2**attempt)
            time.sleep(delay + random.uniform(0, delay))
    raise RuntimeError("unreachable")  # pragma: no cover


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
