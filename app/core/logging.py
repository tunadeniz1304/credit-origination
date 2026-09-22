"""Logging setup: console plus a rotating file handler under ``logs/``.

The :func:`get_logger` factory is idempotent: each named logger gets its
handlers installed exactly once, so repeated imports or agent construction
never duplicate log lines.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from app.core.config import PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "credit_agent.log"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 1_000_000
_BACKUP_COUNT = 3


def get_logger(name: str = "credit_agent", level: int = logging.INFO) -> logging.Logger:
    """Return a console + rotating-file logger, configured once per name."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured; idempotency guard
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    os.makedirs(_LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = True
    return logger
