"""Notification outbox for asynchronous decision delivery.

Headed for LOS integrations, the outbox pattern captures every terminal
decision as a durable, replayable notification entry. Entries are persisted
as JSONL (append-only) and can later be dispatched by a webhook/email worker
that marks each entry delivered.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import Settings

_write_lock = threading.Lock()

EVENT_APPROVED = "APPLICATION_APPROVED"
EVENT_REJECTED = "APPLICATION_REJECTED"
EVENT_FAILED = "APPLICATION_FAILED"


class NotificationEntry(BaseModel):
    """One durable outbox entry awaiting delivery."""

    id: str
    application_id: str
    event: str
    channel: str = "webhook"
    payload: dict = Field(default_factory=dict)
    timestamp: str
    delivered: bool = False


def _outbox_dir(settings: Settings) -> Path:
    return settings.result_dir / "outbox"


def enqueue_notification(
    application_id: str,
    event: str,
    payload: dict | None = None,
    settings: Settings | None = None,
) -> NotificationEntry:
    """Append one notification entry to the outbox and return it."""
    from app.core.config import get_settings

    settings = settings or get_settings()
    entry = NotificationEntry(
        id=f"ntf-{uuid.uuid4().hex[:12]}",
        application_id=application_id,
        event=event,
        payload=payload or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    outbox = _outbox_dir(settings)
    outbox.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with open(outbox / "notifications.jsonl", "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
    return entry


def list_notifications(
    limit: int = 50,
    settings: Settings | None = None,
) -> list[NotificationEntry]:
    """Return undelivered-first outbox entries (newest first)."""
    from app.core.config import get_settings

    settings = settings or get_settings()
    path = _outbox_dir(settings) / "notifications.jsonl"
    if not path.exists():
        return []
    entries: list[NotificationEntry] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(NotificationEntry.model_validate_json(line))
            except ValueError:
                continue
    entries.reverse()
    entries.sort(key=lambda e: (e.delivered, e.timestamp))
    return entries[: max(1, min(limit, 200))]


def mark_delivered(
    notification_id: str,
    settings: Settings | None = None,
) -> bool:
    """Mark one notification entry delivered; ``False`` when unknown."""
    from app.core.config import get_settings

    settings = settings or get_settings()
    path = _outbox_dir(settings) / "notifications.jsonl"
    if not path.exists():
        return False
    with _write_lock:
        lines: list[str] = []
        rewrites: list[str] = []
        found = False
        with open(path, "r", encoding="utf-8") as fh:
            lines = [line.rstrip("\n") for line in fh]
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = NotificationEntry.model_validate_json(line)
            except ValueError:
                rewrites.append(line)
                continue
            if entry.id == notification_id and not entry.delivered:
                entry.delivered = True
                found = True
            rewrites.append(entry.model_dump_json())
        if found:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(rewrites) + "\n")
    return found


def dispatch_pending(settings: Settings | None = None) -> int:
    """Simulate webhook delivery: mark every pending entry delivered.

    Returns the number of entries delivered in this pass.
    """
    from app.core.config import get_settings

    settings = settings or get_settings()
    dispatched = 0
    for entry in list_notifications(limit=200, settings=settings):
        if not entry.delivered:
            if mark_delivered(entry.id, settings):
                dispatched += 1
    return dispatched

