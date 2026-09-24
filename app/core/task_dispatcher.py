"""Worker-agnostic task dispatch.

The pipeline must run the same across environments: under Docker it runs
real Celery tasks on a Redis broker; locally (no Redis) and in the test
suite it runs the identical task body inline. This module hides that
choice behind one ``enqueue`` call.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.worker.tasks import INLINE_TASKS, _celery


class TaskReceipt(BaseModel):
    """Result of dispatching one task."""

    backend: str  # "celery" | "inline"
    task_name: str
    task_id: str
    status: str  # "QUEUED" for celery, "COMPLETED" for inline
    result: Any | None = None


class UnknownTaskError(KeyError):
    """Raised when a task name is not registered on the active backend."""


def resolve_backend(settings: Settings | None = None) -> str:
    """Decide the active backend: explicit setting wins; auto pings Redis."""
    settings = settings or get_settings()
    requested = settings.task_queue_backend
    if requested in ("celery", "inline"):
        return requested
    # auto-detection: prefer Celery only when the broker is reachable.
    try:
        import redis as redis_module

        client = redis_module.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
        client.ping()
        return "celery"
    except Exception:  # redis unreachable/absent -> inline fallback
        return "inline"


def run_coroutine_safe(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine without clashing with an already-running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TaskDispatcher:
    """Enqueues work by name on the active backend (celery or inline)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = resolve_backend(self.settings)
        self.logger = get_logger("task.dispatcher")
        self.logger.info("TaskDispatcher active backend: %s", self.backend)

    def enqueue(self, task_name: str, **kwargs: Any) -> TaskReceipt:
        """Dispatch ``task_name`` with kwargs; honor the active backend."""
        if self.backend == "celery":
            return self._dispatch_celery(task_name, **kwargs)
        return self._dispatch_inline(task_name, **kwargs)

    def _dispatch_celery(self, task_name: str, **kwargs: Any) -> TaskReceipt:
        celery_task = _celery.tasks.get(task_name)
        if celery_task is None:
            raise UnknownTaskError(task_name)
        async_result = celery_task.delay(**kwargs)
        self.logger.info("Enqueued %s on celery broker -> task_id=%s", task_name, async_result.id)
        return TaskReceipt(
            backend="celery",
            task_name=task_name,
            task_id=async_result.id,
            status="QUEUED",
        )

    def _dispatch_inline(self, task_name: str, **kwargs: Any) -> TaskReceipt:
        fn = INLINE_TASKS.get(task_name)
        if fn is None:
            raise UnknownTaskError(task_name)
        fake_id = f"inline-{uuid.uuid4().hex}"
        self.logger.info("Running %s inline (no broker)", task_name)
        value = fn(**kwargs)
        return TaskReceipt(
            backend="inline",
            task_name=task_name,
            task_id=fake_id,
            status="COMPLETED",
            result=value,
        )
