"""Engine and session factory.

One engine per database URL (cached). PostgreSQL uses the standard pool.

SQLite (local/inline mode) allows a single writer. v1 let every session
start writing and serialised only the audit append with a separate process
lock, so two threads could each hold one lock and wait for the other until
``busy_timeout`` expired ("database is locked" → HTTP 500). Now:

* WAL journal + ``synchronous=NORMAL`` (readers never block the writer);
* explicit ``BEGIN`` via SQLAlchemy events, so SAVEPOINTs work (the audit
  append retries inside a savepoint);
* one **application-level writer lock**: a writer session acquires it when
  its root transaction begins (before its first statement) and releases it
  when that transaction ends. Taking it at the start matters in WAL mode: a
  deferred transaction that read first cannot upgrade to a write once another
  writer has committed (``SQLITE_BUSY_SNAPSHOT`` fails immediately, without
  waiting). Sessions marked ``info["readonly"]`` (GET requests) skip it and
  take it only if they flush. All writes of this process are serialised in
  one fixed lock order, which also keeps the audit hash chain linear. Write
  transactions must stay short: slow work (PDFs, LLM text, document analysis)
  runs outside them.

``init_db`` creates tables for SQLite/dev (``DB_AUTO_CREATE``); Docker runs
``alembic upgrade head``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


class WriterLockTimeout(RuntimeError):
    """The single SQLite writer slot was not free in time (surfaced as HTTP 503)."""


# A plain Lock (not RLock): FastAPI may finish a request's session on another thread
# than the one that flushed, and only non-owned locks can be released cross-thread.
_writer_lock = threading.Lock()


def _acquire_writer(session: Session, *_: Any) -> None:
    if session.info.get("writer_lock"):
        return
    bind = session.get_bind()
    if bind.dialect.name != "sqlite":
        return
    timeout = get_settings().db_writer_lock_timeout_seconds
    if not session.info.get("writer_listener"):
        # Registered per session, so sessions not built by our factory release too.
        event.listen(session, "after_transaction_end", _release_writer)
        session.info["writer_listener"] = True
    if not _writer_lock.acquire(timeout=timeout):
        raise WriterLockTimeout(f"SQLite writer busy for more than {timeout:.0f}s")
    session.info["writer_lock"] = True


def _begin_writer(session: Session, transaction: Any) -> None:
    if transaction.parent is None and not session.info.get("readonly"):
        _acquire_writer(session)


def _after_commit(session: Session) -> None:
    # A route that commits mid-request and keeps reading (then hands work to a
    # background task) must not re-take the writer slot for those reads.
    session.info["readonly"] = True


def _release_writer(session: Session, transaction: Any) -> None:
    if transaction.parent is not None:  # savepoints do not end the unit of work
        return
    if session.info.pop("writer_lock", False):
        _writer_lock.release()


def writer_lock_held() -> bool:
    return _writer_lock.locked()


# Hooks run after a unit of work commits (e.g. flushing buffered LLM call records
# in their own short transaction once the caller's write lock is released).
_after_commit_hooks: list[Callable[[], None]] = []
_hook_guard = threading.local()


def register_after_commit(hook: Callable[[], None]) -> None:
    if hook not in _after_commit_hooks:
        _after_commit_hooks.append(hook)


def _run_hooks() -> None:
    if getattr(_hook_guard, "active", False):
        return
    _hook_guard.active = True
    try:
        for hook in list(_after_commit_hooks):
            hook()
    finally:
        _hook_guard.active = False


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    if url.startswith("sqlite"):
        raw = url.split("sqlite:///", 1)[-1]
        if raw and raw != ":memory:":
            Path(raw).parent.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        busy_ms = settings.db_busy_timeout_ms
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": busy_ms / 1000},
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            future=True,
        )

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - driver hook
            # Let SQLAlchemy emit BEGIN itself (pysqlite's implicit transactions break
            # SAVEPOINT semantics).
            dbapi_conn.isolation_level = None
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={busy_ms}")
            cursor.close()

        @event.listens_for(engine, "begin")
        def _begin(conn) -> None:  # pragma: no cover - driver hook
            conn.exec_driver_sql("BEGIN")

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20, future=True)


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    return _engine_for(settings.sqlalchemy_url)


@lru_cache(maxsize=4)
def _factory(url: str) -> sessionmaker[Session]:
    factory = sessionmaker(bind=_engine_for(url), expire_on_commit=False, future=True)
    if url.startswith("sqlite"):
        event.listen(factory, "after_transaction_create", _begin_writer)
        event.listen(factory, "before_flush", _acquire_writer)
        event.listen(factory, "after_commit", _after_commit)
    return factory


def session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    settings = settings or get_settings()
    return _factory(settings.sqlalchemy_url)


@contextmanager
def session_scope(settings: Settings | None = None, *, readonly: bool = False) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = session_factory(settings)()
    session.info["readonly"] = readonly
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    _run_hooks()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session committed at request end."""
    with session_scope() as session:
        yield session


def init_db(settings: Settings | None = None) -> None:
    from app.db.models import Base

    settings = settings or get_settings()
    if settings.db_auto_create:
        Base.metadata.create_all(get_engine(settings))
