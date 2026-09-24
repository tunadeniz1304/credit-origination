"""Hand an application to the active worker backend.

Celery (Docker): the task is queued on Redis and the HTTP request returns at
once. Inline (no Redis): the pipeline runs as a FastAPI background task after
the response is sent, so requests are no longer blocked for the whole
pipeline (bug #16). The resolved backend is cached per process.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import Application

logger = get_logger("workflow.dispatch")


@lru_cache(maxsize=1)
def active_backend() -> str:
    from app.core.task_dispatcher import resolve_backend

    return resolve_backend()


def dispatch_processing(
    session: Session, app: Application, background: BackgroundTasks | None = None
) -> str:
    backend = active_backend()
    if backend == "celery":
        from app.core.task_dispatcher import TaskDispatcher

        receipt = TaskDispatcher().enqueue("app.tasks.process_application", application_id=app.id)
        app.task_id, app.queue_backend = receipt.task_id, "celery"
        session.commit()
        return backend
    from app.worker.tasks import INLINE_TASKS

    app.task_id, app.queue_backend = f"inline-{uuid.uuid4().hex[:12]}", "inline"
    session.commit()
    task = INLINE_TASKS["app.tasks.process_application"]
    if background is not None:
        background.add_task(task, application_id=app.id)
    else:
        task(application_id=app.id)
    return backend
