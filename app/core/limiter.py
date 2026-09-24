"""Shared ``slowapi`` rate limiter (keyed by client address)."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[get_settings().rate_limit_default],
    enabled=get_settings().rate_limit_enabled,
    headers_enabled=False,
)
