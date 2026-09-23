"""Docker deployment topology tests (no docker daemon required).

These guard the compose/worker contract: three services wired to Redis,
healthy-start ordering, the shared image with distinct commands, and the
Celery app entrypoint the worker resolves. ``docker compose config`` lint is
the operational companion check (see README).
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    with open(ROOT / "docker-compose.yml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_compose_defines_three_services():
    services = _compose()["services"]
    assert set(services) == {"app", "worker", "redis"}


def test_app_and_worker_rely_on_healthy_redis():
    services = _compose()["services"]
    for name in ("app", "worker"):
        depends = services[name]["depends_on"]["redis"]
        assert depends == {"condition": "service_healthy"}
    healthcheck = services["redis"].get("healthcheck")
    assert healthcheck and "redis-cli" in healthcheck["test"]

    worker_cmd = services["worker"]["command"]
    assert worker_cmd.startswith("celery -A app.worker.celery_app:celery_app worker")


def test_app_and_worker_share_image_and_celery_backend():
    services = _compose()["services"]
    assert services["app"]["build"] == services["worker"]["build"] == "."
    for name in ("app", "worker"):
        environment = services[name]["environment"]
        assert environment["TASK_QUEUE_BACKEND"] == "celery"
        assert environment["REDIS_URL"] == "redis://redis:6379/0"


def test_dockerfile_runs_uvicorn_on_port_8000():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert 'CMD ["uvicorn", "app.main:app"' in dockerfile


def test_dockerfile_installs_pinned_requirements():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements.txt ." in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile


def test_celery_app_entrypoint_autodiscovers_tasks():
    """The worker entrypoint must register app.tasks.process_application."""
    from app.worker.celery_app import celery_app

    assert celery_app.main == "credit_agent"
    registered = set(celery_app.tasks.keys())
    assert "app.tasks.process_application" in registered
    assert "app.tasks.ping" in registered
