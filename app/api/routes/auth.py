"""Authentication: login, applicant self-registration, current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, db_session
from app.api.schemas import LoginRequest, RegisterRequest
from app.core.captcha import verify_captcha
from app.core.config import get_settings
from app.core.limiter import limiter
from app.core.security import (
    ROLE_LABELS,
    AuthError,
    Principal,
    create_access_token,
    csrf_token_for,
    hash_password,
    revoke_token,
    verify_password,
)
from app.db.audit import append_audit
from app.db.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _token_response(user: User, response: Response) -> dict:
    """Set the HttpOnly session cookie + CSRF cookie; the body keeps a Bearer token for API clients."""
    settings = get_settings()
    principal = Principal(
        user_id=user.id, username=user.username, role=user.role, full_name=user.full_name
    )
    token = create_access_token(principal)
    csrf = csrf_token_for(token)
    max_age = settings.jwt_expire_minutes * 60
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )
    response.set_cookie(
        settings.csrf_cookie_name,
        csrf,
        max_age=max_age,
        httponly=False,  # the page reads it and echoes it in X-CSRF-Token
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )
    return {
        "access_token": token,
        "csrf_token": csrf,
        "token_type": "bearer",
        "role": user.role,
        "role_label": ROLE_LABELS.get(user.role, user.role),
        "username": user.username,
        "full_name": user.full_name,
        "expires_in_minutes": get_settings().jwt_expire_minutes,
    }


@router.post("/login")
@limiter.limit(get_settings().rate_limit_login)
def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    session: Session = Depends(db_session),
) -> dict:
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
    return _token_response(user, response)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    settings = get_settings()
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else None
    token = token or request.cookies.get(settings.session_cookie_name)
    if token:
        try:
            revoke_token(token)  # the JWT stops working now, not at its expiry
        except AuthError as exc:  # shared revocation list unreachable: say so, do not 500
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "oturum şu an kapatılamadı, tekrar deneyin"
            ) from exc
    for name in (settings.session_cookie_name, settings.csrf_cookie_name):
        response.delete_cookie(name, path="/", secure=settings.cookie_secure, samesite="strict")
    return {"status": "ok"}


@router.get("/config")
def auth_config() -> dict:
    """What the login screen may offer (demo shortcuts only in demo mode)."""
    settings = get_settings()
    return {
        "demo_mode": settings.demo_mode,
        "registration_enabled": settings.registration_enabled,
        "captcha": settings.captcha_provider != "none",
    }


@router.post("/register", status_code=201)
@limiter.limit(get_settings().rate_limit_register)
def register(
    request: Request,
    response: Response,
    body: RegisterRequest,
    session: Session = Depends(db_session),
) -> dict:
    settings = get_settings()
    if not settings.registration_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "yeni kayıt kapalı")
    client_ip = request.client.host if request.client else None
    if not verify_captcha(body.captcha_token, client_ip):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "CAPTCHA doğrulaması başarısız")
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
    return _token_response(user, response)


@router.get("/me")
def me(user: Principal = Depends(current_user)) -> dict:
    return {
        **user.model_dump(),
        "role_label": ROLE_LABELS.get(user.role, user.role),
        "is_staff": user.is_staff,
    }
