"""FastAPI dependencies: database session, authentication, RBAC, ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import STAFF_ROLES, AuthError, Principal, csrf_valid, decode_token
from app.db.models import Application
from app.db.session import session_scope

_bearer = HTTPBearer(auto_error=False)


def db_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session


SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def current_user(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)
) -> Principal:
    """Bearer token (API clients) or the HttpOnly session cookie (browser).

    Cookie-authenticated unsafe requests must echo the signed CSRF token in
    the ``X-CSRF-Token`` header (double submit); Bearer requests are not
    exposed to CSRF because browsers never attach the header automatically.
    """
    settings = get_settings()
    token: str | None = None
    if credentials is not None and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    else:
        token = request.cookies.get(settings.session_cookie_name)
        csrf = request.headers.get(settings.csrf_header_name)
        if token and request.method not in SAFE_METHODS and not csrf_valid(token, csrf):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF doğrulaması başarısız")
    if not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "kimlik doğrulama gerekli",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(token)
    except AuthError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "geçersiz veya süresi dolmuş oturum",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_roles(*roles: str) -> Callable[[Principal], Principal]:
    allowed = set(roles) | {"admin"}

    def dependency(user: Principal = Depends(current_user)) -> Principal:
        if user.role not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "bu işlem için yetkiniz yok")
        return user

    return dependency


STAFF = tuple(STAFF_ROLES)
require_staff = require_roles(*STAFF)


def load_application(session: Session, application_id: str, user: Principal) -> Application:
    app = session.get(Application, application_id)
    if app is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "başvuru bulunamadı")
    if user.role == "basvuran" and app.owner_user_id != user.user_id:
        # Do not reveal existence of other applicants' records.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "başvuru bulunamadı")
    if user.role not in STAFF_ROLES and user.role != "basvuran":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bu işlem için yetkiniz yok")
    return app
