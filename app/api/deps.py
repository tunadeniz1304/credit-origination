"""FastAPI dependencies: database session, authentication, RBAC, ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import STAFF_ROLES, AuthError, Principal, decode_token
from app.db.models import Application
from app.db.session import session_scope

_bearer = HTTPBearer(auto_error=False)


def db_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "kimlik doğrulama gerekli",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
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
