"""Persisted application + result store (two-phase JSON files).

The API submits an application and persists it immediately to
``settings.result_dir``; a separate Celery worker (or the inline dispatcher)
later loads the full application by id, runs the pipeline, and persists the
outcome. Reads fall back from result files to application files, and the API
layer falls back to the in-memory store, so GET stays coherent in both
worker topologies.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.core.config import Settings
from app.models import LoanApplication, PipelineResult


def generate_application_id() -> str:
    """Return a human-friendly, globally unique application id."""
    return f"APP-{uuid.uuid4().hex[:12].upper()}"


def _result_path(application_id: str, settings: Settings) -> Path:
    return settings.result_dir / f"{application_id}.result.json"


def _application_path(application_id: str, settings: Settings) -> Path:
    return settings.result_dir / f"{application_id}.application.json"


def persist_application(
    application_id: str,
    application: LoanApplication,
    settings: Settings,
) -> Path:
    """Write the submitted application to disk; return its path."""
    path = _application_path(application_id, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(application.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_application(
    application_id: str,
    settings: Settings,
) -> LoanApplication | None:
    """Read a persisted application back; ``None`` when absent or corrupt."""
    path = _application_path(application_id, settings)
    if not path.exists():
        return None
    try:
        return LoanApplication.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def persist_result(
    application_id: str,
    result: PipelineResult,
    settings: Settings,
) -> Path:
    """Write the pipeline result to disk; return its path."""
    path = _result_path(application_id, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_result(
    application_id: str,
    settings: Settings,
) -> PipelineResult | None:
    """Read a persisted result back; ``None`` when absent or corrupt."""
    path = _result_path(application_id, settings)
    if not path.exists():
        return None
    try:
        return PipelineResult.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def persist_error(application_id: str, message: str, settings: Settings) -> Path:
    """Record an unrecoverable processing error for an application."""
    path = settings.result_dir / f"{application_id}.error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"error": message}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
