"""CAPTCHA hook for public endpoints (self-registration).

``CAPTCHA_PROVIDER=none`` (default) accepts every request. ``hook`` calls the
verifier registered with :func:`register_verifier` — plug a real provider
(hCaptcha, reCAPTCHA, Turnstile) there without touching the route. Without a
registered verifier ``hook`` fails closed.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings, get_settings

Verifier = Callable[[str | None, str | None], bool]  # (token, client ip) -> ok
_verifier: Verifier | None = None


def register_verifier(verifier: Verifier | None) -> None:
    global _verifier
    _verifier = verifier


def verify_captcha(
    token: str | None, client_ip: str | None, settings: Settings | None = None
) -> bool:
    settings = settings or get_settings()
    if settings.captcha_provider == "none":
        return True
    if _verifier is None:
        return False
    return bool(_verifier(token, client_ip))
