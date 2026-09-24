"""Lightweight append-only audit trail (JSONL) for pipeline events.

Regulated credit platforms must record every meaningful transition. Each
audit entry is one JSONL line under ``settings.result_dir / "audit"`` with a
UTC timestamp, an uppercase machine-readable action and a human detail
string. Writes are appends under a lock, so the trail stays append-only.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from app.core.config import Settings

_write_lock = threading.Lock()

# Canonical lifecycle actions.
ACTION_SUBMITTED = "APPLICATION_SUBMITTED"
ACTION_QUEUED = "APPLICATION_QUEUED"
ACTION_PROCESSING = "APPLICATION_PROCESSING"
ACTION_APPROVED = "APPLICATION_APPROVED"
ACTION_REJECTED = "APPLICATION_REJECTED"
ACTION_FAILED = "APPLICATION_FAILED"
ACTION_DOCUMENT_DELIVERED = "DOCUMENT_DELIVERED"
ACTION_REPROCESSED = "APPLICATION_REPROCESSED"


class AuditEntry(BaseModel):
    """One immutable audit line."""

    application_id: str
    action: str
    detail: str = ""
    timestamp: str  # ISO-8601 UTC


def _audit_dir(settings: Settings) -> Path:
    return settings.result_dir / "audit"


def append_audit(
    application_id: str,
    action: str,
    detail: str = "",
    settings: Settings | None = None,
) -> Path:
    """Append one audit entry and return the trail file path."""
    from app.core.config import get_settings

    settings = settings or get_settings()
    entry = AuditEntry(
        application_id=application_id,
        action=action,
        detail=detail,
        timestamp=datetime.now(UTC).isoformat(),
    )
    audit_dir = _audit_dir(settings)
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{application_id}.jsonl"
    with _write_lock, open(path, "a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")
    return path


def load_audit(application_id: str, settings: Settings | None = None) -> list[AuditEntry]:
    """Read the audit trail for one application (chronological order)."""
    from app.core.config import get_settings

    settings = settings or get_settings()
    path = _audit_dir(settings) / f"{application_id}.jsonl"
    if not path.exists():
        return []
    entries: list[AuditEntry] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(AuditEntry.model_validate_json(line))
            except ValueError:
                continue  # tolerate a torn append
    return entries
