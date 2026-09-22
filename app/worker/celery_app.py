"""Celery application factory for the credit operations worker.

The broker/backend point at Redis in the Docker topology. Constructing the
application is side-effect free: no connection is opened here, so imports
succeed even without a live Redis (tests and local demos use the inline
task dispatcher instead).
"""
from __future__ import annotations

from celery import Celery

from app.core.config import get_settings


def create_celery_app(
    broker_url: str | None = None,
    result_backend: str | None = None,
) -> Celery:
    """Build a configured Celery application bound to Redis."""
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
    )
    app.autodiscover_tasks(["app.worker"])
    return app


celery_app = create_celery_app()
