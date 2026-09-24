"""Browser session security: HttpOnly cookie, signed double-submit CSRF, prod defaults."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import captcha
from app.core.config import Settings, get_settings
from app.core.security import csrf_token_for, csrf_valid
from app.main import app
from tests.helpers import DEMO_LOGIN


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, username: str = "basvuran") -> dict:
    response = client.post(
        "/api/v1/auth/login", json={"username": username, "password": DEMO_LOGIN}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_login_sets_httponly_session_and_readable_csrf_cookie(client):
    body = _login(client)
    cookies = client.cookies
    settings = get_settings()
    assert cookies.get(settings.session_cookie_name)
    assert cookies.get(settings.csrf_cookie_name) == body["csrf_token"]
    headers = client.post(
        "/api/v1/auth/login", json={"username": "basvuran", "password": DEMO_LOGIN}
    ).headers.get_list("set-cookie")
    session_cookie = next(h for h in headers if h.startswith(settings.session_cookie_name))
    csrf_cookie = next(h for h in headers if h.startswith(settings.csrf_cookie_name))
    assert "HttpOnly" in session_cookie and "samesite=strict" in session_cookie.lower()
    assert "HttpOnly" not in csrf_cookie


def test_cookie_session_reads_without_csrf_but_writes_need_it(client):
    body = _login(client)
    me = client.get("/api/v1/auth/me")  # no Authorization header: cookie only
    assert me.status_code == 200 and me.json()["username"] == "basvuran"
    assert client.post("/api/v1/applications", json={}).status_code == 403
    forged = client.post("/api/v1/applications", json={}, headers={"X-CSRF-Token": "x" * 64})
    assert forged.status_code == 403
    ok = client.post("/api/v1/applications", json={}, headers={"X-CSRF-Token": body["csrf_token"]})
    assert ok.status_code == 422  # passed auth + CSRF, rejected only by validation


def test_bearer_clients_are_not_subject_to_csrf(client):
    token = _login(client)["access_token"]
    client.cookies.clear()
    response = client.post(
        "/api/v1/applications", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422


def test_logout_clears_cookies(client):
    _login(client)
    client.post("/api/v1/auth/logout")
    assert client.get("/api/v1/auth/me").status_code == 401


def test_csrf_token_is_bound_to_the_session():
    assert csrf_valid("session-a", csrf_token_for("session-a"))
    assert not csrf_valid("session-b", csrf_token_for("session-a"))
    assert not csrf_valid("session-a", None)


def test_registration_switch_and_captcha_hook(client, monkeypatch):
    settings = get_settings()
    body = {"username": "yeni_kullanici_1", "password": "Guclu123!", "full_name": "Yeni Kişi"}
    monkeypatch.setattr(settings, "registration_enabled", False)
    assert client.post("/api/v1/auth/register", json=body).status_code == 403
    monkeypatch.setattr(settings, "registration_enabled", True)
    monkeypatch.setattr(settings, "captcha_provider", "hook")
    captcha.register_verifier(None)
    assert client.post("/api/v1/auth/register", json=body).status_code == 400  # fails closed
    captcha.register_verifier(lambda token, ip: token == "ok")
    try:
        assert client.post("/api/v1/auth/register", json=body).status_code == 400
        created = client.post("/api/v1/auth/register", json={**body, "captcha_token": "ok"})
        assert created.status_code == 201 and created.json()["csrf_token"]
    finally:
        captcha.register_verifier(None)
    config = client.get("/api/v1/auth/config").json()
    assert config == {"demo_mode": True, "registration_enabled": True, "captcha": True}


def test_prod_environment_defaults(monkeypatch):
    monkeypatch.delenv("COOKIE_SECURE", raising=False)  # the suite runs over plain http
    prod = Settings(_env_file=None, app_env="prod")  # type: ignore[call-arg]
    assert prod.seed_demo_users is False and prod.demo_mode is False
    forced = Settings(_env_file=None, app_env="prod", SEED_DEMO_USERS=True)  # type: ignore[call-arg]
    assert forced.seed_demo_users is True
    assert Settings(_env_file=None).cookie_secure is True


def test_prod_start_does_not_seed_demo_users(tmp_path):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.core.users import ensure_demo_users
    from app.db.models import Base, User

    engine = create_engine(f"sqlite:///{tmp_path / 'prod.db'}")
    Base.metadata.create_all(engine)
    prod = Settings(_env_file=None, app_env="prod")  # type: ignore[call-arg]
    with Session(engine) as session:
        assert ensure_demo_users(session, prod) == 0
        assert session.execute(select(User)).first() is None


def test_health_reports_ocr_and_demo_mode(client):
    body = client.get("/health").json()
    assert set(body["ocr"]) >= {"available", "reason"} and body["demo_mode"] is True
