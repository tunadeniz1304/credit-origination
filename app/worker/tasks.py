"""Celery tasks and their inline (broker-less) equivalents.

Each task body is a plain function registered both as a Celery task and in
``INLINE_TASKS`` so :class:`~app.core.task_dispatcher.TaskDispatcher` runs the
identical code without Redis (local development, tests). All state lives in
the database, so the API and any number of workers stay coherent.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.worker.celery_app import create_celery_app

_celery = create_celery_app()
logger = get_logger("worker.tasks")


def _ping() -> str:
    return "pong"


def _process_application(application_id: str) -> dict[str, Any]:
    """Advance one application through the staged pipeline."""
    from app.core.rules import load_workflow
    from app.core.task_dispatcher import run_coroutine_safe
    from app.workflow.pipeline import Pipeline
    from app.workflow.states import State

    lock = _application_lock(application_id)
    if lock is not None and not lock.acquire(blocking=False):
        return {"application_id": application_id, "skipped": "already being processed"}
    try:
        state = run_coroutine_safe(Pipeline(get_settings()).run(application_id))
    except KeyError:
        return {"application_id": application_id, "error": "not found"}
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:  # lock expired
                logger.debug("application lock already released", exc_info=True)
    if state == State.VERI_TOPLANIYOR.value:
        _schedule_retry(application_id, load_workflow().data_collection_retry_seconds)
    return {"application_id": application_id, "state": state}


def _application_lock(application_id: str) -> Any:
    """Cross-process lock so one application is never processed twice at once."""
    from app.core.task_dispatcher import resolve_backend

    if resolve_backend() != "celery":
        return None
    import redis

    client = redis.Redis.from_url(get_settings().redis_url)
    return client.lock(f"anil2:process:{application_id}", timeout=900)


def _schedule_retry(application_id: str, countdown: int) -> None:
    from app.core.task_dispatcher import resolve_backend

    if resolve_backend() != "celery":
        return  # inline: retried by retry_stalled / reprocess
    from app.core.rules import load_workflow
    from app.db.models import Application
    from app.db.session import session_scope

    with session_scope() as session:
        app = session.get(Application, application_id)
        if app is None or app.retry_count > load_workflow().data_collection_max_retries:
            return
    process_application.apply_async(kwargs={"application_id": application_id}, countdown=countdown)


def _dispatch_notifications() -> dict[str, Any]:
    from app.db.outbox import dispatch_pending

    return dispatch_pending()


def _retry_stalled() -> dict[str, Any]:
    """Re-run applications stuck in data collection (provider outages)."""
    from app.db.models import Application
    from app.db.session import session_scope
    from app.workflow.service import ApplicationService
    from app.workflow.states import State

    with session_scope() as session:
        ids = list(
            session.execute(
                select(Application.id).where(Application.state == State.VERI_TOPLANIYOR.value)
            ).scalars()
        )
        # Documents completed while the gate was writing its letter (upload race).
        service = ApplicationService(session)
        for app in session.execute(
            select(Application).where(Application.state == State.BELGE_BEKLENIYOR.value)
        ).scalars():
            if not service.missing_documents(app):
                ids.append(app.id)
    from app.core.task_dispatcher import resolve_backend

    if resolve_backend() == "celery":
        for app_id in ids:  # fan out; never block the beat-driven task on the LLM
            process_application.delay(application_id=app_id)
        return {"retried": len(ids), "queued": ids}
    results = [_process_application(app_id) for app_id in ids]
    return {"retried": len(ids), "results": results}


def _apply_retention() -> dict[str, Any]:
    """KVKK retention: anonymise PII of rejected/cancelled applications after N days."""
    from app.db.models import Applicant, Application, utcnow
    from app.db.session import session_scope
    from app.workflow.states import State

    settings = get_settings()
    cutoff = utcnow() - timedelta(days=settings.retention_days_rejected)
    closed = (State.OTOMATIK_RET.value, State.REDDEDILDI.value, State.IPTAL.value)
    anonymised = 0
    with session_scope() as session:
        rows = session.execute(
            select(Applicant)
            .join(Application, Application.applicant_id == Applicant.id)
            .where(
                Application.state.in_(closed),
                Application.updated_at < cutoff,
                Applicant.anonymized_at.is_(None),
            )
        ).scalars()
        for applicant in rows:
            applicant.name_enc = applicant.phone_enc = applicant.email_enc = None
            applicant.address_enc = applicant.iban_enc = applicant.tckn_enc = None
            applicant.phone_bidx = applicant.iban_bidx = applicant.address_bidx = None
            applicant.anonymized_at = utcnow()
            anonymised += 1
    logger.info("retention job anonymised %d applicants", anonymised)
    return {"anonymised": anonymised, "cutoff": cutoff.isoformat()}


def _compute_drift() -> dict[str, Any]:
    from app.db.session import session_scope
    from app.governance.drift import compute_drift_report

    with session_scope() as session:
        return compute_drift_report(session)


ping = _celery.task(name="app.tasks.ping")(_ping)
process_application = _celery.task(name="app.tasks.process_application")(_process_application)
dispatch_notifications = _celery.task(name="app.tasks.dispatch_notifications")(
    _dispatch_notifications
)
retry_stalled = _celery.task(name="app.tasks.retry_stalled")(_retry_stalled)
apply_retention = _celery.task(name="app.tasks.apply_retention")(_apply_retention)
compute_drift = _celery.task(name="app.tasks.compute_drift")(_compute_drift)

INLINE_TASKS: dict[str, Callable[..., Any]] = {
    "app.tasks.ping": _ping,
    "app.tasks.process_application": _process_application,
    "app.tasks.dispatch_notifications": _dispatch_notifications,
    "app.tasks.retry_stalled": _retry_stalled,
    "app.tasks.apply_retention": _apply_retention,
    "app.tasks.compute_drift": _compute_drift,
}
