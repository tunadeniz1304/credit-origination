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


def test_logout_revokes_the_bearer_token(client):
    token = client.post(
        "/api/v1/auth/login", json={"username": "uzman", "password": "Demo123!"}
    ).json()["access_token"]
    client.cookies.clear()
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200
    client.post("/api/v1/auth/logout", headers=headers)
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_prod_refuses_unsafe_settings():
    strong = "x" * 40
    unsafe = Settings(_env_file=None, app_env="prod")  # type: ignore[call-arg]
    problems = " ".join(unsafe.production_problems())
    assert "JWT_SECRET" in problems and "PII_ENCRYPTION_KEY" in problems
    assert unsafe.registration_enabled is False  # self-registration off by default in prod
    safe = Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="prod",
        jwt_secret=strong,
        pii_encryption_key=strong,
        blind_index_key=strong,
        session_revocation_backend="redis",
    )
    assert safe.production_problems() == []
    open_registration = Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="prod",
        jwt_secret=strong,
        pii_encryption_key=strong,
        blind_index_key=strong,
        registration_enabled=True,
    )
    assert any("CAPTCHA" in p for p in open_registration.production_problems())


def test_app_does_not_start_with_unsafe_prod_settings(monkeypatch):
    import asyncio

    from app.main import app, lifespan

    monkeypatch.setattr(
        "app.main.get_settings",
        lambda: Settings(_env_file=None, app_env="prod"),  # type: ignore[call-arg]
    )

    async def start():
        async with lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="unsafe production settings"):
        asyncio.run(start())


def _strong_prod(**overrides) -> Settings:
    strong = "x" * 40
    values = {
        "app_env": "prod",
        "jwt_secret": strong,
        "pii_encryption_key": strong,
        "blind_index_key": strong,
        "session_revocation_backend": "redis",
    }
    return Settings(_env_file=None, **{**values, **overrides})  # type: ignore[arg-type]


def test_prod_refuses_demo_users_and_a_local_revocation_list():
    """Audit v2.1 round 2: demo users could be forced on; logout held in one worker only."""
    assert any(
        "SEED_DEMO_USERS" in p for p in _strong_prod(SEED_DEMO_USERS=True).production_problems()
    )
    local = _strong_prod(session_revocation_backend="auto").production_problems()
    assert any("SESSION_REVOCATION_BACKEND" in p for p in local)


def test_worker_refuses_unsafe_production_settings(monkeypatch):
    from app.worker import celery_app

    monkeypatch.setattr(
        celery_app,
        "get_settings",
        lambda: Settings(_env_file=None, app_env="prod"),  # type: ignore[call-arg]
    )
    with pytest.raises(RuntimeError, match="worker: unsafe production settings"):
        celery_app._refuse_unsafe_production()
    monkeypatch.setattr(celery_app, "get_settings", _strong_prod)
    celery_app._refuse_unsafe_production()  # safe settings start


def test_crypto_never_uses_the_development_key_in_prod():
    from app.core.crypto import blind_index, encrypt

    no_keys = _strong_prod(pii_encryption_key="", blind_index_key="")
    with pytest.raises(RuntimeError, match="PII_ENCRYPTION_KEY"):
        encrypt("12345678901", no_keys)
    with pytest.raises(RuntimeError, match="BLIND_INDEX_KEY"):
        blind_index("12345678901", no_keys)


class _BrokenRedis:
    def ping(self):
        return True

    def exists(self, key):
        raise ConnectionError("redis went away")

    def set(self, *args, **kwargs):
        raise ConnectionError("redis went away")


def test_revocation_list_fails_closed(monkeypatch):
    from app.core.security import AuthError, _RevocationList

    required = _strong_prod()
    revoked = _RevocationList()

    def unreachable(*args, **kwargs):
        raise ConnectionError("no redis")

    monkeypatch.setattr("redis.Redis.from_url", unreachable)
    with pytest.raises(AuthError, match="unavailable"):  # required Redis down: refuse, not accept
        revoked.contains("jti-1", required)
    monkeypatch.setattr("redis.Redis.from_url", lambda *a, **k: _BrokenRedis())
    with pytest.raises(AuthError, match="unavailable"):  # error mid-check: refuse, not 500
        revoked.contains("jti-1", required)
    with pytest.raises(AuthError):
        revoked.add("jti-2", 9e9, required)
    monkeypatch.setattr("redis.Redis.from_url", unreachable)
    assert revoked.contains("jti-2", required) is True  # revoked here: answered without Redis
    auto = Settings(_env_file=None, session_revocation_backend="auto")  # type: ignore[call-arg]
    assert revoked.contains("jti-2", auto) is True  # kept locally although Redis failed


def test_revocation_list_reprobes_redis_in_auto_mode(monkeypatch):
    import fakeredis

    from app.core.security import _RevocationList

    auto = Settings(_env_file=None, session_revocation_backend="auto")  # type: ignore[call-arg]
    revoked = _RevocationList()
    monkeypatch.setattr(
        "redis.Redis.from_url", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
    )
    assert revoked.contains("x", auto) is False  # Redis down: local list only
    shared = fakeredis.FakeRedis()
    shared.set("anil2:revoked:x", 1)
    monkeypatch.setattr("redis.Redis.from_url", lambda *a, **k: shared)
    assert revoked.contains("x", auto) is False  # not re-probed before the retry interval
    revoked._next_probe = 0.0  # interval elapsed
    assert revoked.contains("x", auto) is True  # Redis is back: its list counts again


def test_token_stops_working_when_the_user_is_deactivated_or_changes_role(client):
    from sqlalchemy import select, update

    from app.db.models import User
    from app.db.session import session_scope

    token = _login(client, "uzman2")["access_token"]
    client.cookies.clear()
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200
    with session_scope() as session:
        user = session.execute(select(User).where(User.username == "uzman2")).scalar_one()
        original_role = user.role
        session.execute(update(User).where(User.id == user.id).values(role="basvuran"))
    try:
        assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
        with session_scope() as session:
            session.execute(
                update(User)
                .where(User.username == "uzman2")
                .values(role=original_role, active=False)
            )
        assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
    finally:
        with session_scope() as session:
            session.execute(
                update(User)
                .where(User.username == "uzman2")
                .values(role=original_role, active=True)
            )
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200
