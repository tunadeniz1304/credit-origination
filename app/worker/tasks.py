"""Celery tasks and their inline (broker-less) equivalents.

Each domain task is implemented once as a plain function; it is registered
both as a Celery task (``@celery_app.task``) and in ``INLINE_TASKS`` so the
:class:`TaskDispatcher` can run the exact same body synchronously when no
Redis broker is available (local development and the test suite).
"""
from __future__ import annotations

from typing import Any, Callable

from app.api.store import update_record_status
from app.core.config import get_settings
from app.engine.pipeline import ApplicationPipeline
from app.engine.store import (
    load_application,
    persist_error,
    persist_result,
)
from app.models import ApplicationStatus
from app.worker.celery_app import create_celery_app

# Dispatcher-local Celery app (worker process uses the shared one).
_celery = create_celery_app()


@_celery.task(name="app.tasks.ping")
def ping() -> str:
    """Broker connectivity smoke task: always returns ``"pong"``."""
    return "pong"


def _inline_ping() -> str:
    return "pong"


@_celery.task(name="app.tasks.process_application")
def process_application(application_id: str) -> dict[str, Any]:
    """Run the full credit pipeline for a persisted application.

    Loads the application from disk (the API process persisted it at submit
    time), runs the async pipeline, persists result + BDDK report files and
    mirrors the status into the in-memory record store when present.
    """
    settings = get_settings()
    from app.core.task_dispatcher import run_coroutine_safe
    from app.engine.audit import (
        ACTION_APPROVED,
        ACTION_FAILED,
        ACTION_PROCESSING,
        ACTION_REJECTED,
        append_audit,
    )

    application = load_application(application_id, settings)
    if application is None:
        persist_error(application_id, "application record not found", settings)
        append_audit(application_id, ACTION_FAILED, "application record missing", settings)
        update_record_status(application_id, status=ApplicationStatus.FAILED)
        return {
            "application_id": application_id,
            "status": ApplicationStatus.FAILED.value,
            "error": "application record not found",
        }
    update_record_status(application_id, status=ApplicationStatus.PROCESSING)
    append_audit(application_id, ACTION_PROCESSING, "worker started pipeline", settings)
    try:
        result = run_coroutine_safe(
            ApplicationPipeline(settings).run(
                application, application_id=application_id
            )
        )
        persist_result(application_id, result, settings)
        update_record_status(application_id, status=result.status, result=result)
        action = (
            ACTION_APPROVED if result.status == ApplicationStatus.APPROVED else ACTION_REJECTED
        )
        append_audit(application_id, action, f"committee verdict: {result.status.value}", settings)
        from app.engine.notifier import enqueue_notification

        enqueue_notification(
            application_id,
            event=result.status.value,
            payload={"status": result.status.value, "amount": result.decision.suggested_amount}
            if result.decision
            else {"status": result.status.value},
            settings=settings,
        )
        return result.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - record failure, never crash the caller
        persist_error(application_id, str(exc), settings)
        append_audit(application_id, ACTION_FAILED, str(exc), settings)
        update_record_status(
            application_id, status=ApplicationStatus.FAILED, error=str(exc)
        )
        return {
            "application_id": application_id,
            "status": ApplicationStatus.FAILED.value,
            "error": str(exc),
        }


def _inline_process_application(application_id: str) -> dict[str, Any]:
    return process_application(application_id)


@_celery.task(name="app.tasks.dispatch_notifications")
def dispatch_notifications() -> dict[str, Any]:
    """Worker task: drain the notification outbox (simulated webhooks)."""
    from app.engine.notifier import dispatch_pending

    dispatched = dispatch_pending()
    return {"dispatched": dispatched}


def _inline_dispatch_notifications() -> dict[str, Any]:
    return dispatch_notifications()


# Callables available to the inline task dispatcher (name -> function).
INLINE_TASKS: dict[str, Callable[..., Any]] = {
    "app.tasks.ping": _inline_ping,
    "app.tasks.process_application": _inline_process_application,
    "app.tasks.dispatch_notifications": _inline_dispatch_notifications,
}
