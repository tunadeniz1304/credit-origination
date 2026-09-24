"""Docker deployment topology tests (no docker daemon required)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    with open(ROOT / "docker-compose.yml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_compose_services():
    services = _compose()["services"]
    assert {"postgres", "redis", "app", "worker", "beat"} <= set(services)


def test_app_and_worker_share_image_env_and_env_file():
    services = _compose()["services"]
    for name in ("app", "worker"):
        env = services[name]["environment"]
        assert env["TASK_QUEUE_BACKEND"] == "celery"
        assert env["DATABASE_URL"].startswith("postgresql+psycopg://")
        assert services[name]["env_file"][0]["path"] == ".env"
        assert services[name]["build"] == "."
    assert services["app"]["depends_on"]["postgres"] == {"condition": "service_healthy"}
    assert services["worker"]["command"].startswith(
        "celery -A app.worker.celery_app:celery_app worker"
    )


def test_healthchecks_defined():
    services = _compose()["services"]
    for name in ("postgres", "redis", "worker"):
        assert services[name].get("healthcheck")


def test_dockerfile_contract():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "alembic upgrade head" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "COPY .env" not in dockerfile  # secrets are injected via env_file only


def test_env_is_never_shipped():
    assert ".env" in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_celery_app_registers_tasks():
    from app.worker.celery_app import celery_app

    registered = set(celery_app.tasks.keys())
    assert {
        "app.tasks.process_application",
        "app.tasks.dispatch_notifications",
        "app.tasks.retry_stalled",
    } <= registered


def test_alembic_migration_matches_models(tmp_path, monkeypatch):
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from alembic import command
    from app.db.models import Base

    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "alembic"))
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
    tables = set(inspect(create_engine(url)).get_table_names())
    assert set(Base.metadata.tables) <= tables
