"""Celery application factory for the credit operations worker.

Broker/backend point at Redis in the Docker topology. Construction opens no
connection, so imports succeed without Redis (tests and local runs use the
inline dispatcher). Beat schedules the outbox dispatcher, the stalled
application retry, the KVKK retention job and the drift monitor.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings


def create_celery_app(broker_url: str | None = None, result_backend: str | None = None) -> Celery:
    settings = get_settings()
    app = Celery(
        "credit_agent",
        broker=broker_url or settings.redis_url,
        backend=result_backend or settings.result_backend_url,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Europe/Istanbul",
        enable_utc=True,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        result_expires=3600,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        beat_schedule={
            "dispatch-outbox": {"task": "app.tasks.dispatch_notifications", "schedule": 15.0},
            "retry-stalled": {"task": "app.tasks.retry_stalled", "schedule": 60.0},
            "kvkk-retention": {"task": "app.tasks.apply_retention", "schedule": 86_400.0},
            "drift-monitor": {"task": "app.tasks.compute_drift", "schedule": 3_600.0},
        },
    )
    app.autodiscover_tasks(["app.worker"])
    return app


celery_app = create_celery_app()
