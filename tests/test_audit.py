"""Audit trail module tests."""
from __future__ import annotations

from app.core.config import Settings
from app.engine.audit import (
    ACTION_APPROVED,
    ACTION_SUBMITTED,
    append_audit,
    load_audit,
)


def _settings(tmp_path) -> Settings:
    return Settings(
        report_output_dir=str(tmp_path / "reports"),
        result_store_dir=str(tmp_path / "results"),
        upload_dir=str(tmp_path / "uploads"),
    )


def test_audit_appends_and_loads_chronologically(tmp_path):
    settings = _settings(tmp_path)
    append_audit("APP-X", ACTION_SUBMITTED, "accepted", settings)
    append_audit("APP-X", ACTION_APPROVED, "verdict APPROVED", settings)
    entries = load_audit("APP-X", settings)
    assert [e.action for e in entries] == [ACTION_SUBMITTED, ACTION_APPROVED]
    assert entries[1].detail == "verdict APPROVED"
    assert entries[0].application_id == "APP-X"


def test_audit_is_append_only_and_isolated_per_application(tmp_path):
    settings = _settings(tmp_path)
    for _ in range(3):
        append_audit("APP-A", ACTION_SUBMITTED, "", settings)
    append_audit("APP-B", ACTION_SUBMITTED, "", settings)
    assert len(load_audit("APP-A", settings)) == 3
    assert len(load_audit("APP-B", settings)) == 1
    assert load_audit("APP-NOPE", settings) == []


def test_audit_timestamps_are_utc_iso(tmp_path):
    settings = _settings(tmp_path)
    append_audit("APP-X", ACTION_SUBMITTED, "", settings)
    entry = load_audit("APP-X", settings)[0]
    assert entry.timestamp.endswith("+00:00") or "T" in entry.timestamp
