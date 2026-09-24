"""Authentication (JWT) and role-based access control.

Roles: ``basvuran`` (own applications only), ``uzman``, ``kidemli_uzman``,
``komite``, ``model_yoneticisi`` and ``admin``. Passwords are hashed with
PBKDF2-HMAC-SHA256; tokens are short-lived HS256 JWTs signed with
``JWT_SECRET``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt
from pydantic import BaseModel

from app.core.config import Settings, get_settings

Role = Literal["basvuran", "uzman", "kidemli_uzman", "komite", "model_yoneticisi", "admin"]
ROLES: tuple[str, ...] = (
    "basvuran",
    "uzman",
    "kidemli_uzman",
    "komite",
    "model_yoneticisi",
    "admin",
)
STAFF_ROLES: frozenset[str] = frozenset({"uzman", "kidemli_uzman", "komite", "admin"})
CREDIT_AUTHORITY_RANK: dict[str, int] = {"uzman": 1, "kidemli_uzman": 2, "komite": 3, "admin": 3}
ROLE_LABELS: dict[str, str] = {
    "basvuran": "Başvuran",
    "uzman": "Krediler Uzmanı",
    "kidemli_uzman": "Kıdemli Krediler Uzmanı",
    "komite": "Kredi Komitesi",
    "model_yoneticisi": "Model Yöneticisi",
    "admin": "Sistem Yöneticisi",
}
_ITERATIONS = 120_000


class Principal(BaseModel):
    """The authenticated caller."""

    user_id: str
    username: str
    role: str
    full_name: str = ""

    @property
    def is_staff(self) -> bool:
        return self.role in STAFF_ROLES

    @property
    def authority_rank(self) -> int:
        return CREDIT_AUTHORITY_RANK.get(self.role, 0)


def hash_password(password: str, *, iterations: int = _ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = encoded.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iterations)
    )
    return hmac.compare_digest(digest, base64.b64decode(digest_b64))


def create_access_token(principal: Principal, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": principal.user_id,
        "username": principal.username,
        "role": principal.role,
        "name": principal.full_name,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
        "iss": "anil2",
    }
    return jwt.encode(
        claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )


class AuthError(Exception):
    """Invalid or expired credentials."""


def decode_token(token: str, settings: Settings | None = None) -> Principal:
    settings = settings or get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer="anil2",
        )
    except jwt.PyJWTError as exc:
        raise AuthError(type(exc).__name__) from exc
    if claims.get("role") not in ROLES:
        raise AuthError("unknown role")
    return Principal(
        user_id=claims["sub"],
        username=claims["username"],
        role=claims["role"],
        full_name=claims.get("name", ""),
    )


def csrf_token_for(session_token: str, settings: Settings | None = None) -> str:
    """Signed double-submit token bound to the session cookie (resists cookie tossing)."""
    settings = settings or get_settings()
    key = settings.jwt_secret.get_secret_value().encode("utf-8")
    return hmac.new(key, b"csrf:" + session_token.encode("utf-8"), hashlib.sha256).hexdigest()


def csrf_valid(session_token: str, presented: str | None, settings: Settings | None = None) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(csrf_token_for(session_token, settings), presented)
