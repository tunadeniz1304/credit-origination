"""Secure, per-application document storage.

Uploads are size-limited, type-checked by **magic bytes** (the client's
content type is never trusted), stored under
``data/uploads/{application_id}/{uuid}_{code}{ext}`` with a sanitised name,
hashed with SHA-256 and optionally scanned by ClamAV. One application's
documents can never overwrite or leak into another's.
"""

from __future__ import annotations

import hashlib
import re
import socket
import struct
import uuid
from pathlib import Path

from pydantic import BaseModel

from app.core.config import Settings, get_settings

MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"%PDF-", "application/pdf", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
)


class UploadRejected(ValueError):
    """The upload violates size/type/malware policy."""


class StoredFile(BaseModel):
    path: str
    filename: str
    sha256: str
    mime: str
    size: int


def detect_mime(data: bytes) -> str | None:
    for magic, mime, _ in MAGIC:
        if data.startswith(magic):
            return mime
    try:
        data[:4096].decode("utf-8")
    except UnicodeDecodeError:
        return None
    return "text/plain" if b"\x00" not in data[:4096] else None


def sanitise_filename(name: str) -> str:
    stem = Path(name or "belge").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "belge"
    return stem[:80]


def _extension(mime: str) -> str:
    for _, candidate, ext in MAGIC:
        if candidate == mime:
            return ext
    return ".txt"


def clamav_scan(data: bytes, host: str, port: int = 3310) -> bool:  # pragma: no cover - optional
    """INSTREAM scan against a ClamAV daemon; True when clean."""
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(b"zINSTREAM\0")
        for offset in range(0, len(data), 8192):
            chunk = data[offset : offset + 8192]
            sock.sendall(struct.pack("!L", len(chunk)) + chunk)
        sock.sendall(struct.pack("!L", 0))
        reply = sock.recv(4096)
    return b"OK" in reply and b"FOUND" not in reply


def store_upload(
    application_id: str,
    code: str,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
) -> StoredFile:
    settings = settings or get_settings()
    if not re.fullmatch(r"APP-[A-F0-9]{12}", application_id):
        raise UploadRejected("invalid application id")
    if not data:
        raise UploadRejected("empty file")
    if len(data) > settings.max_upload_bytes:
        raise UploadRejected(f"file exceeds {settings.max_upload_mb:g} MB limit")
    mime = detect_mime(data)
    if mime is None or mime not in settings.allowed_upload_types:
        raise UploadRejected("unsupported file type")
    if settings.clamav_host and not clamav_scan(data, settings.clamav_host):
        raise UploadRejected("malware detected")
    directory = settings.uploads_dir / application_id
    directory.mkdir(parents=True, exist_ok=True)
    safe = sanitise_filename(filename)
    target = directory / f"{uuid.uuid4().hex[:12]}_{code.upper()}{_extension(mime)}"
    target.write_bytes(data)
    return StoredFile(
        path=str(target),
        filename=safe,
        sha256=hashlib.sha256(data).hexdigest(),
        mime=mime,
        size=len(data),
    )
