"""Worker tasks and the backend-agnostic dispatcher.

The suite runs with ``TASK_QUEUE_BACKEND=inline``; the Celery branches are
exercised with fake Celery task objects (``.delay`` / ``.apply_async``) and
``fakeredis`` locks instead of a real broker.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update

from app.core import task_dispatcher as td
from app.core.config import get_settings
from app.core.rules import load_workflow
from app.db.models import Applicant, Application, OutboxMessage, utcnow
from app.db.outbox import enqueue
from app.db.session import session_scope
from app.main import app
from app.worker import tasks
from app.workflow.pipeline import Pipeline
from app.workflow.states import State
from tests.helpers import login, payload, submit_complete


# ------------------------------------------------------------------ fakes
class FakeAsyncResult:
    def __init__(self) -> None:
        self.id = f"celery-{uuid.uuid4().hex}"


class FakeCeleryTask:
    """Records ``delay`` / ``apply_async`` calls instead of talking to a broker."""

    def __init__(self) -> None:
        self.delayed: list[dict] = []
        self.scheduled: list[tuple[dict, int]] = []

    def delay(self, **kwargs):
        self.delayed.append(kwargs)
        return FakeAsyncResult()

    def apply_async(self, kwargs=None, countdown=None):
        self.scheduled.append((kwargs, countdown))
        return FakeAsyncResult()


@pytest.fixture
def celery_mode(monkeypatch):
    """Force the celery branch and route Redis to one shared in-memory server."""
    server = fakeredis.FakeServer()
    monkeypatch.setattr(td, "resolve_backend", lambda settings=None: "celery")
    monkeypatch.setattr(
        "redis.Redis.from_url",
        lambda *a, **k: fakeredis.FakeRedis(server=server),
    )
    fake = FakeCeleryTask()
    monkeypatch.setattr(tasks, "process_application", fake)
    # The dispatcher looks tasks up in the Celery registry: route it to the fake as well, so an
    # application submitted through the API never reaches a real broker or result backend.
    monkeypatch.setattr(
        td, "_celery", SimpleNamespace(tasks={"app.tasks.process_application": fake})
    )
    return SimpleNamespace(server=server, task=fake)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def _app_started(client):
    """Tests run in random order: the app lifespan creates the schema first."""
    return client


# Tests run in random order (pytest-randomly): the app lifespan creates the schema.
pytestmark = pytest.mark.usefixtures("client")


@pytest.fixture(scope="module")
def applicant_headers(client):
    return login(client, "basvuran")


def _submit_bare(client, headers) -> str:
    """Submit an application without documents (it parks in BELGE_BEKLENIYOR)."""
    response = client.post("/api/v1/applications", json=payload("temiz"), headers=headers)
    assert response.status_code == 202, response.text
    return response.json()["application_id"]


def _force(application_id: str, **values) -> None:
    with session_scope() as session:
        session.execute(
            update(Application).where(Application.id == application_id).values(**values)
        )


def _state(application_id: str) -> str:
    with session_scope() as session:
        return session.get(Application, application_id).state


# ------------------------------------------------------------------ resolve_backend
@pytest.mark.parametrize("backend", ["celery", "inline"])
def test_resolve_backend_explicit_setting_wins(backend, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("explicit backend must not ping redis")

    monkeypatch.setattr("redis.Redis.from_url", boom)
    settings = get_settings().model_copy(update={"task_queue_backend": backend})
    assert td.resolve_backend(settings) == backend


def test_resolve_backend_auto_uses_celery_when_redis_answers(monkeypatch):
    seen = {}

    def from_url(url, **kwargs):
        seen["url"], seen["kwargs"] = url, kwargs
        return fakeredis.FakeRedis()

    monkeypatch.setattr("redis.Redis.from_url", from_url)
    settings = get_settings().model_copy(
        update={"task_queue_backend": "auto", "redis_url": "redis://broker:6379/3"}
    )
    assert td.resolve_backend(settings) == "celery"
    assert seen["url"] == "redis://broker:6379/3"
    assert seen["kwargs"]["socket_connect_timeout"] == 0.5


def test_resolve_backend_auto_falls_back_to_inline_when_redis_is_down(monkeypatch):
    class DeadRedis:
        def ping(self):
            raise ConnectionError("connection refused")

    monkeypatch.setattr("redis.Redis.from_url", lambda *a, **k: DeadRedis())
    settings = get_settings().model_copy(update={"task_queue_backend": "auto"})
    assert td.resolve_backend(settings) == "inline"


def test_resolve_backend_defaults_to_global_settings():
    assert td.resolve_backend() == "inline"  # conftest pins TASK_QUEUE_BACKEND=inline


# ------------------------------------------------------------------ run_coroutine_safe
async def _answer(value):
    await asyncio.sleep(0)
    return value * 2


def test_run_coroutine_safe_without_running_loop():
    assert td.run_coroutine_safe(_answer(21)) == 42


def test_run_coroutine_safe_uses_private_loop_when_one_is_reported(monkeypatch):
    """With a loop reported as running it must not call ``asyncio.run`` (which would raise)."""
    loops = []
    real_new_loop = asyncio.new_event_loop

    def tracking_new_loop():
        loop = real_new_loop()
        loops.append(loop)
        return loop

    monkeypatch.setattr(td.asyncio, "get_running_loop", lambda: object())
    monkeypatch.setattr(td.asyncio, "new_event_loop", tracking_new_loop)

    def no_asyncio_run(coro):
        coro.close()
        raise AssertionError("asyncio.run must not be used inside a running loop")

    monkeypatch.setattr(td.asyncio, "run", no_asyncio_run)
    assert td.run_coroutine_safe(_answer(5)) == 10
    assert len(loops) == 1 and loops[0].is_closed()


def test_run_coroutine_safe_closes_private_loop_on_error(monkeypatch):
    loops = []
    real_new_loop = asyncio.new_event_loop

    def tracking_new_loop():
        loop = real_new_loop()
        loops.append(loop)
        return loop

    async def fails():
        raise ValueError("pipeline exploded")

    monkeypatch.setattr(td.asyncio, "get_running_loop", lambda: object())
    monkeypatch.setattr(td.asyncio, "new_event_loop", tracking_new_loop)
    with pytest.raises(ValueError, match="pipeline exploded"):
        td.run_coroutine_safe(fails())
    assert loops[0].is_closed()


# ------------------------------------------------------------------ TaskDispatcher
def test_dispatcher_runs_inline_task_and_returns_completed_receipt():
    dispatcher = td.TaskDispatcher()
    assert dispatcher.backend == "inline"
    receipt = dispatcher.enqueue("app.tasks.ping")
    assert receipt.backend == "inline"
    assert receipt.status == "COMPLETED"
    assert receipt.result == "pong"
    assert receipt.task_id.startswith("inline-")
    assert dispatcher.enqueue("app.tasks.ping").task_id != receipt.task_id


def test_dispatcher_inline_forwards_kwargs(monkeypatch):
    calls = []
    monkeypatch.setitem(td.INLINE_TASKS, "app.tasks.echo", lambda **kw: calls.append(kw) or kw)
    receipt = td.TaskDispatcher().enqueue("app.tasks.echo", application_id="A1", n=2)
    assert calls == [{"application_id": "A1", "n": 2}]
    assert receipt.result == {"application_id": "A1", "n": 2}


def test_dispatcher_inline_unknown_task_raises():
    with pytest.raises(td.UnknownTaskError) as excinfo:
        td.TaskDispatcher().enqueue("app.tasks.does_not_exist")
    assert isinstance(excinfo.value, KeyError)
    assert "app.tasks.does_not_exist" in str(excinfo.value)


def test_dispatcher_celery_queues_via_delay(monkeypatch):
    fake = FakeCeleryTask()
    monkeypatch.setattr(
        td, "_celery", SimpleNamespace(tasks={"app.tasks.process_application": fake})
    )
    settings = get_settings().model_copy(update={"task_queue_backend": "celery"})
    dispatcher = td.TaskDispatcher(settings)
    assert dispatcher.backend == "celery"
    receipt = dispatcher.enqueue("app.tasks.process_application", application_id="APP-1")
    assert fake.delayed == [{"application_id": "APP-1"}]
    assert receipt.backend == "celery"
    assert receipt.status == "QUEUED"
    assert receipt.result is None
    assert receipt.task_id.startswith("celery-")


def test_dispatcher_celery_unknown_task_raises(monkeypatch):
    monkeypatch.setattr(td, "_celery", SimpleNamespace(tasks={}))
    settings = get_settings().model_copy(update={"task_queue_backend": "celery"})
    with pytest.raises(td.UnknownTaskError):
        td.TaskDispatcher(settings).enqueue("app.tasks.process_application", application_id="X")


def test_every_inline_task_is_registered_on_celery():
    for name, fn in tasks.INLINE_TASKS.items():
        assert tasks._celery.tasks[name].name == name
        assert callable(fn)


# ------------------------------------------------------------------ process_application
def test_process_application_unknown_id_reports_not_found():
    assert tasks._process_application("NOPE-404") == {
        "application_id": "NOPE-404",
        "error": "not found",
    }


def test_process_application_inline_advances_complete_application(client, applicant_headers):
    application_id = submit_complete(client, applicant_headers, "temiz")
    result = tasks._process_application(application_id)
    assert result["application_id"] == application_id
    assert result["state"] == _state(application_id)
    assert result["state"] not in (State.BELGE_BEKLENIYOR.value, State.VERI_TOPLANIYOR.value)


def test_process_application_skips_when_lock_is_held(celery_mode, monkeypatch):
    monkeypatch.setattr(Pipeline, "run", pytest.fail)  # must never reach the pipeline
    holder = fakeredis.FakeRedis(server=celery_mode.server).lock("anil2:process:APP-L", timeout=60)
    assert holder.acquire(blocking=False)
    try:
        result = tasks._process_application("APP-L")
    finally:
        holder.release()
    assert result == {"application_id": "APP-L", "skipped": "already being processed"}


def test_process_application_releases_lock_after_run(celery_mode, monkeypatch):
    async def fake_run(self, application_id):
        return State.UZMAN_INCELEMESI.value

    monkeypatch.setattr(Pipeline, "run", fake_run)
    assert tasks._process_application("APP-R")["state"] == State.UZMAN_INCELEMESI.value
    redis_client = fakeredis.FakeRedis(server=celery_mode.server)
    assert not redis_client.exists("anil2:process:APP-R")
    assert celery_mode.task.scheduled == []  # only VERI_TOPLANIYOR schedules a retry


def test_process_application_releases_lock_on_not_found(celery_mode):
    result = tasks._process_application("MISSING-CELERY")
    assert result["error"] == "not found"
    assert not fakeredis.FakeRedis(server=celery_mode.server).exists("anil2:process:MISSING-CELERY")


def test_process_application_tolerates_expired_lock(monkeypatch):
    class ExpiredLock:
        def acquire(self, blocking):
            return True

        def release(self):
            raise RuntimeError("lock expired")

    async def fake_run(self, application_id):
        return State.OTOMATIK_ONAY.value

    monkeypatch.setattr(tasks, "_application_lock", lambda application_id: ExpiredLock())
    monkeypatch.setattr(Pipeline, "run", fake_run)
    assert tasks._process_application("APP-E") == {
        "application_id": "APP-E",
        "state": State.OTOMATIK_ONAY.value,
    }


def test_application_lock_is_none_inline_and_namespaced_on_celery(celery_mode, monkeypatch):
    lock = tasks._application_lock("APP-9")
    assert lock.name == "anil2:process:APP-9"
    assert lock.timeout == 900
    monkeypatch.setattr(td, "resolve_backend", lambda settings=None: "inline")
    assert tasks._application_lock("APP-9") is None


def test_data_collection_state_schedules_celery_retry(
    celery_mode, monkeypatch, client, applicant_headers
):
    application_id = _submit_bare(client, applicant_headers)

    async def stalled(self, app_id):
        return State.VERI_TOPLANIYOR.value

    monkeypatch.setattr(Pipeline, "run", stalled)
    result = tasks._process_application(application_id)
    assert result["state"] == State.VERI_TOPLANIYOR.value
    assert celery_mode.task.scheduled == [
        ({"application_id": application_id}, load_workflow().data_collection_retry_seconds)
    ]


def test_schedule_retry_gives_up_after_max_retries(celery_mode, client, applicant_headers):
    application_id = _submit_bare(client, applicant_headers)
    _force(application_id, retry_count=load_workflow().data_collection_max_retries + 1)
    tasks._schedule_retry(application_id, 30)
    assert celery_mode.task.scheduled == []
    _force(application_id, retry_count=0)
    tasks._schedule_retry(application_id, 30)
    assert celery_mode.task.scheduled == [({"application_id": application_id}, 30)]


def test_schedule_retry_ignores_unknown_application(celery_mode):
    tasks._schedule_retry("GHOST-1", 30)
    assert celery_mode.task.scheduled == []


def test_schedule_retry_is_noop_inline(monkeypatch):
    fake = FakeCeleryTask()
    monkeypatch.setattr(tasks, "process_application", fake)
    tasks._schedule_retry("ANY", 30)
    assert fake.scheduled == []


# ------------------------------------------------------------------ retry_stalled
@pytest.fixture
def stalled_ids(client, applicant_headers):
    """One app stuck in data collection, one whose docs landed while it waited, one truly waiting."""
    in_collection = _submit_bare(client, applicant_headers)
    _force(in_collection, state=State.VERI_TOPLANIYOR.value)
    raced = submit_complete(client, applicant_headers, "temiz")
    _force(raced, state=State.BELGE_BEKLENIYOR.value)
    waiting = _submit_bare(client, applicant_headers)
    assert _state(waiting) == State.BELGE_BEKLENIYOR.value
    return SimpleNamespace(in_collection=in_collection, raced=raced, waiting=waiting)


def test_retry_stalled_fans_out_on_celery(celery_mode, stalled_ids):
    result = tasks._retry_stalled()
    queued = [call["application_id"] for call in celery_mode.task.delayed]
    assert result["queued"] == queued
    assert result["retried"] == len(queued)
    assert stalled_ids.in_collection in queued
    assert stalled_ids.raced in queued
    assert stalled_ids.waiting not in queued  # still missing documents


def test_retry_stalled_processes_inline(monkeypatch, stalled_ids):
    processed = []

    def fake_process(application_id):
        processed.append(application_id)
        return {"application_id": application_id, "state": "X"}

    monkeypatch.setattr(tasks, "_process_application", fake_process)
    result = tasks._retry_stalled()
    assert result["retried"] == len(processed)
    assert [r["application_id"] for r in result["results"]] == processed
    assert {stalled_ids.in_collection, stalled_ids.raced} <= set(processed)
    assert stalled_ids.waiting not in processed


def test_retry_stalled_task_via_dispatcher_moves_raced_application_forward(
    client, applicant_headers
):
    raced = submit_complete(client, applicant_headers, "temiz")
    _force(raced, state=State.BELGE_BEKLENIYOR.value)
    receipt = td.TaskDispatcher().enqueue("app.tasks.retry_stalled")
    ids = [r["application_id"] for r in receipt.result["results"]]
    assert raced in ids
    assert _state(raced) != State.BELGE_BEKLENIYOR.value


# ------------------------------------------------------------------ notifications
@pytest.fixture
def isolated_outbox():
    """Park other modules' pending messages so this test neither sends nor counts them.

    The key of the message the test creates is yielded; it is deleted afterwards
    because ``/notifications`` lists SENT rows first and test_api expects none.
    """
    key = f"test-dispatch-{uuid.uuid4().hex}"
    with session_scope() as session:
        parked = list(
            session.execute(
                select(OutboxMessage.id).where(OutboxMessage.status == "PENDING")
            ).scalars()
        )
        session.execute(
            update(OutboxMessage).where(OutboxMessage.id.in_(parked)).values(status="PARKED")
        )
    yield key
    with session_scope() as session:
        session.execute(delete(OutboxMessage).where(OutboxMessage.idempotency_key == key))
        session.execute(
            update(OutboxMessage).where(OutboxMessage.id.in_(parked)).values(status="PENDING")
        )


def test_dispatch_notifications_delivers_pending_outbox(isolated_outbox):
    key = isolated_outbox
    with session_scope() as session:
        enqueue(
            session,
            event="test.event",
            aggregate_id="AGG-1",
            payload={"k": "v"},
            channel="console",
            idempotency_key=key,
        )
    assert tasks._dispatch_notifications() == {"sent": 1, "failed": 0}
    with session_scope() as session:
        message = session.execute(
            select(OutboxMessage).where(OutboxMessage.idempotency_key == key)
        ).scalar_one()
        assert message.status == "SENT"
        assert message.attempts == 1
    # idempotent: a second run never re-sends the delivered message
    assert tasks._dispatch_notifications() == {"sent": 0, "failed": 0}
    with session_scope() as session:
        message = session.execute(
            select(OutboxMessage).where(OutboxMessage.idempotency_key == key)
        ).scalar_one()
        assert message.attempts == 1


# ------------------------------------------------------------------ retention
def test_retention_anonymises_only_old_closed_applications(client, applicant_headers):
    old_closed = _submit_bare(client, applicant_headers)
    recent_closed = _submit_bare(client, applicant_headers)
    old_open = _submit_bare(client, applicant_headers)
    long_ago = utcnow() - timedelta(days=get_settings().retention_days_rejected + 30)
    _force(old_closed, state=State.IPTAL.value, updated_at=long_ago)
    _force(recent_closed, state=State.REDDEDILDI.value)
    _force(old_open, updated_at=long_ago)

    result = tasks._apply_retention()
    assert result["anonymised"] >= 1
    assert "cutoff" in result

    with session_scope() as session:

        def applicant(app_id):
            return session.get(Applicant, session.get(Application, app_id).applicant_id)

        gone = applicant(old_closed)
        assert gone.anonymized_at is not None
        for field in (
            "name_enc",
            "phone_enc",
            "email_enc",
            "address_enc",
            "iban_enc",
            "tckn_enc",
            "phone_bidx",
            "iban_bidx",
            "address_bidx",
        ):
            assert getattr(gone, field) is None, field
        for kept in (recent_closed, old_open):
            assert applicant(kept).anonymized_at is None
            assert applicant(kept).name_enc is not None

    # already-anonymised applicants are not counted twice
    again = tasks._apply_retention()
    assert again["anonymised"] == 0


# ------------------------------------------------------------------ drift / ping
def test_compute_drift_returns_report(client):
    report = tasks._compute_drift()
    assert isinstance(report, dict) and report
    via_dispatcher = td.TaskDispatcher().enqueue("app.tasks.compute_drift").result
    assert set(via_dispatcher) == set(report)


def test_ping_task():
    assert tasks._ping() == "pong"
    assert tasks.INLINE_TASKS["app.tasks.ping"] is tasks._ping
