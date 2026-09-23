"""Notification outbox module tests."""
from __future__ import annotations

from app.core.config import Settings
from app.engine.notifier import (
    EVENT_APPROVED,
    enqueue_notification,
    list_notifications,
    mark_delivered,
)


def _settings(tmp_path) -> Settings:
    return Settings(
        report_output_dir=str(tmp_path / "reports"),
        result_store_dir=str(tmp_path / "results"),
        upload_dir=str(tmp_path / "uploads"),
    )


def test_enqueue_then_list_returns_pending_first(tmp_path):
    settings = _settings(tmp_path)
    entry = enqueue_notification("APP-X", EVENT_APPROVED,
                                 {"status": "APPROVED"}, settings)
    entries = list_notifications(settings=settings)
    assert len(entries) == 1
    assert entries[0].id == entry.id
    assert entries[0].event == EVENT_APPROVED
    assert entries[0].delivered is False


def test_mark_delivered_flips_flag_once(tmp_path):
    settings = _settings(tmp_path)
    entry = enqueue_notification("APP-X", EVENT_APPROVED, {}, settings)
    assert mark_delivered(entry.id, settings) is True
    assert list_notifications(settings=settings)[0].delivered is True
    assert mark_delivered(entry.id, settings) is False  # already delivered


def test_delivered_entries_sort_behind_pending(tmp_path):
    settings = _settings(tmp_path)
    first = enqueue_notification("APP-A", "APPLICATION_REJECTED", {}, settings)
    second = enqueue_notification("APP-B", EVENT_APPROVED, {}, settings)
    mark_delivered(first.id, settings)
    entries = list_notifications(settings=settings)
    assert [e.application_id for e in entries] == ["APP-B", "APP-A"]
