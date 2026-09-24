"""Engine and session factory.

One engine per database URL (cached). SQLite gets WAL mode and a relaxed
thread check so FastAPI's threadpool, the inline worker and tests share it
safely; PostgreSQL uses the standard pool. ``init_db`` creates tables for
SQLite/dev (``DB_AUTO_CREATE``); Docker runs ``alembic upgrade head``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


@lru_cache(maxsize=4)
def _engine_for(url: str) -> Engine:
    if url.startswith("sqlite"):
        raw = url.split("sqlite:///", 1)[-1]
        if raw and raw != ":memory:":
            Path(raw).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url, connect_args={"check_same_thread": False, "timeout": 30}, future=True
        )

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - driver hook
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20, future=True)


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    return _engine_for(settings.sqlalchemy_url)


@lru_cache(maxsize=4)
def _factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=_engine_for(url), expire_on_commit=False, future=True)


def session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    settings = settings or get_settings()
    return _factory(settings.sqlalchemy_url)


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session committed at request end."""
    with session_scope() as session:
        yield session


def init_db(settings: Settings | None = None) -> None:
    from app.db.models import Base

    settings = settings or get_settings()
    if settings.db_auto_create:
        Base.metadata.create_all(get_engine(settings))
