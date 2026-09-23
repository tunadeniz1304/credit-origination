"""In-memory application records store (thread-safe).

The persisted JSON store in ``app.engine.store`` is the source of truth for
completed results and for cross-process coherence (Celery worker); this
in-memory layer carries queue-status metadata and recent submissions within
the API process.
"""
from __future__ import annotations

import threading
from typing import Any

from app.models import ApplicationRecord, ApplicationStatus, PipelineResult

records: dict[str, ApplicationRecord] = {}
_lock = threading.Lock()


def store_record(record: ApplicationRecord) -> None:
    """Insert or replace a record under the lock."""
    with _lock:
        records[record.application_id] = record


def get_record(application_id: str) -> ApplicationRecord | None:
    """Return the in-memory record; ``None`` when unknown."""
    with _lock:
        return records.get(application_id)


def update_record_status(
    application_id: str,
    *,
    status: ApplicationStatus,
    result: PipelineResult | None = None,
    error: str | None = None,
) -> ApplicationRecord | None:
    """Mutate status/result/error of a record; return it or ``None``."""
    with _lock:
        record = records.get(application_id)
        if record is None:
            return None
        record.status = status
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error
        return record


def list_records(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent records (newest first) as JSON-ready dicts."""
    with _lock:
        ordered = sorted(
            records.values(),
            key=lambda rec: rec.application_id,
            reverse=True,
        )[:limit]
        return [rec.to_dict() for rec in ordered]
