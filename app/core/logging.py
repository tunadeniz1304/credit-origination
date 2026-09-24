"""Logging setup: console + rotating file, PII masking and request ids.

:func:`get_logger` is idempotent: each named logger gets its handlers exactly
once. Every handler carries :class:`~app.agents.redaction.PIIMaskingFilter`
(TCKN / IBAN / phone / e-mail never reach a log in clear text) and a
request-id filter fed by a ``contextvars`` variable that the HTTP middleware
sets. ``LOG_JSON=true`` switches to one JSON object per line.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

from app.core.config import PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "credit_agent.log"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(request_id)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 1_000_000
_BACKUP_COUNT = 3

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Injects the active request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Structured one-line JSON records (structlog-compatible field names)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "event": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _json_enabled() -> bool:
    return os.environ.get("LOG_JSON", "").lower() in ("1", "true", "yes")


def _install_filters(handler: logging.Handler) -> None:
    from app.agents.redaction import PIIMaskingFilter

    handler.addFilter(RequestIdFilter())
    handler.addFilter(PIIMaskingFilter())


def get_logger(name: str = "credit_agent", level: int = logging.INFO) -> logging.Logger:
    """Return a console + rotating-file logger, configured once per name."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured; idempotency guard
        return logger

    logger.setLevel(level)
    formatter: logging.Formatter = (
        JsonFormatter() if _json_enabled() else logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    _install_filters(console)
    logger.addHandler(console)

    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _install_filters(file_handler)
        logger.addHandler(file_handler)
    except OSError:  # read-only filesystem: console logging only
        pass

    logger.propagate = True
    return logger
