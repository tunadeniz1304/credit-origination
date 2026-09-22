"""Celery tasks and their inline (broker-less) equivalents.

Each domain task is implemented once as a plain function; it is registered
both as a Celery task (``@celery_app.task``) and in ``INLINE_TASKS`` so the
:class:`TaskDispatcher` can run the exact same body synchronously when no
Redis broker is available (local development and the test suite).
"""
from __future__ import annotations

from typing import Any, Callable

from app.worker.celery_app import create_celery_app

# Dispatcher-local Celery app (worker process uses the shared one).
_celery = create_celery_app()


@_celery.task(name="app.tasks.ping")
def ping() -> str:
    """Broker connectivity smoke task: always returns ``"pong"``."""
    return "pong"


def _inline_ping() -> str:
    return "pong"


# Callables available to the inline task dispatcher (name -> function).
INLINE_TASKS: dict[str, Callable[..., Any]] = {
    "app.tasks.ping": _inline_ping,
}
