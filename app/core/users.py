"""Demo users (dev/test only by default; ``SEED_DEMO_USERS`` overrides, off in prod)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.security import hash_password
from app.db.models import User

DEMO_USERS: tuple[tuple[str, str, str], ...] = (
    ("basvuran", "Deniz Başvuran", "basvuran"),
    ("basvuran2", "Ece Başvuran", "basvuran"),
    ("uzman", "Mert Uzman", "uzman"),
    ("uzman2", "Selin Uzman", "uzman"),
    ("kidemli", "Burak Kıdemli", "kidemli_uzman"),
    ("komite", "Kredi Komitesi", "komite"),
    ("modelyon", "Zeynep Model", "model_yoneticisi"),
    ("modelyon2", "Can Model", "model_yoneticisi"),
    ("admin", "Sistem Yöneticisi", "admin"),
)


def active_demo_accounts(session: Session) -> list[str]:
    """Demo usernames still active in the database (their password is public)."""
    names = [username for username, _, _ in DEMO_USERS]
    return sorted(
        session.execute(
            select(User.username).where(User.username.in_(names), User.active.is_(True))
        ).scalars()
    )


def ensure_demo_users(session: Session, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    if not settings.seed_demo_users:
        return 0
    existing = set(session.execute(select(User.username)).scalars())
    password = settings.demo_password.get_secret_value()
    created = 0
    for username, full_name, role in DEMO_USERS:
        if username in existing:
            continue
        session.add(
            User(
                username=username,
                full_name=full_name,
                role=role,
                password_hash=hash_password(password),
            )
        )
        created += 1
    if created:
        get_logger("security.users").info("created %d demo users", created)
    return created
