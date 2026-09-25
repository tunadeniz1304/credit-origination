"""Application-level PII encryption (Fernet) and HMAC blind indexes.

TCKN, name, phone, address, e-mail and IBAN are stored encrypted. Equality
search (duplicate / velocity checks) uses a keyed HMAC-SHA256 "blind index"
over a normalised value, so the database never holds the clear identifier.
Keys come from ``PII_ENCRYPTION_KEY`` / ``BLIND_INDEX_KEY``; without them a
deterministic development key is derived and a warning is logged once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

_DEV_MATERIAL = b"anil2-development-only-key-material"


def _derive(label: bytes) -> bytes:
    return hashlib.sha256(_DEV_MATERIAL + label).digest()


@lru_cache(maxsize=4)
def _fernet(raw_key: str) -> Fernet:
    if raw_key:
        return Fernet(raw_key.encode())
    get_logger("security.crypto").warning(
        "PII_ENCRYPTION_KEY not set: using a development key (never use in production)"
    )
    return Fernet(base64.urlsafe_b64encode(_derive(b"fernet")))


def _blind_key(settings: Settings) -> bytes:
    raw = settings.blind_index_key.get_secret_value()
    if not raw and settings.app_env == "prod":
        raise RuntimeError("BLIND_INDEX_KEY is not set")
    return raw.encode() if raw else _derive(b"blind-index")


def _cipher(settings: Settings | None = None) -> Fernet:
    settings = settings or get_settings()
    raw_key = settings.pii_encryption_key.get_secret_value()
    if not raw_key and settings.app_env == "prod":
        # Never fall back to the public development key in production, whatever process this is.
        raise RuntimeError("PII_ENCRYPTION_KEY is not set")
    return _fernet(raw_key)


def encrypt(value: str | None, settings: Settings | None = None) -> str | None:
    if value is None or value == "":
        return value
    return _cipher(settings).encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(token: str | None, settings: Settings | None = None) -> str | None:
    if token is None or token == "":
        return token
    try:
        return _cipher(settings).decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken:
        return None


def normalise(value: str) -> str:
    """Canonical form for blind indexing (case, whitespace, punctuation)."""
    lowered = value.strip().casefold()
    return re.sub(r"[\s\-.()/]+", "", lowered)


def blind_index(value: str | None, settings: Settings | None = None) -> str | None:
    if not value:
        return None
    settings = settings or get_settings()
    digest = hmac.new(_blind_key(settings), normalise(value).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def mask_tckn(tckn: str | None) -> str:
    if not tckn:
        return ""
    return f"{tckn[:2]}*******{tckn[-2:]}"


def mask_iban(iban: str | None) -> str:
    if not iban:
        return ""
    compact = iban.replace(" ", "")
    return f"{compact[:4]} **** **** {compact[-4:]}"


def mask_phone(phone: str | None) -> str:
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    return f"0{digits[-10:-7]} *** ** {digits[-2:]}" if len(digits) >= 10 else "***"
