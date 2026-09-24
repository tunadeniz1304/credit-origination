"""Authentication: login, applicant self-registration, current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, db_session
from app.api.schemas import LoginRequest, RegisterRequest
from app.core.config import get_settings
from app.core.limiter import limiter
from app.core.security import (
    ROLE_LABELS,
    Principal,
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.audit import append_audit
from app.db.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _token_response(user: User) -> dict:
    principal = Principal(
        user_id=user.id, username=user.username, role=user.role, full_name=user.full_name
    )
    return {
        "access_token": create_access_token(principal),
        "token_type": "bearer",
        "role": user.role,
        "role_label": ROLE_LABELS.get(user.role, user.role),
        "username": user.username,
        "full_name": user.full_name,
        "expires_in_minutes": get_settings().jwt_expire_minutes,
    }


@router.post("/login")
@limiter.limit(get_settings().rate_limit_login)
def login(request: Request, body: LoginRequest, session: Session = Depends(db_session)) -> dict:
    user = session.execute(select(User).where(User.username == body.username)).scalar_one_or_none()
    if user is None or not user.active or not verify_password(body.password, user.password_hash):
        append_audit(
            session,
            actor=body.username,
            action="LOGIN_FAILED",
            entity_type="user",
            entity_id=body.username,
        )
        session.commit()  # keep the failed-login audit entry despite the error response
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "kullanıcı adı veya parola hatalı")
    append_audit(
        session, actor=user.username, action="LOGIN", entity_type="user", entity_id=user.id
    )
    return _token_response(user)


@router.post("/register", status_code=201)
@limiter.limit(get_settings().rate_limit_login)
def register(
    request: Request, body: RegisterRequest, session: Session = Depends(db_session)
) -> dict:
    if session.execute(select(User).where(User.username == body.username)).scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, "kullanıcı adı kullanımda")
    user = User(
        username=body.username,
        full_name=body.full_name,
        role="basvuran",
        password_hash=hash_password(body.password),
    )
    session.add(user)
    session.flush()
    append_audit(
        session,
        actor=user.username,
        action="USER_REGISTERED",
        entity_type="user",
        entity_id=user.id,
    )
    return _token_response(user)


@router.get("/me")
def me(user: Principal = Depends(current_user)) -> dict:
    return {
        **user.model_dump(),
        "role_label": ROLE_LABELS.get(user.role, user.role),
        "is_staff": user.is_staff,
    }
