"""User-facing Turkish labels for enum codes (``rules/labels.yaml``).

Raw codes such as ``IHTIYAC`` or ``MAASLI`` are identifiers, not text: letters,
PDFs and the UI always go through :func:`label`. Unknown codes fall back to the
code itself so a missing entry is visible rather than silently blank.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.rules import read_yaml


@lru_cache(maxsize=1)
def labels() -> dict[str, dict[str, str]]:
    data: dict[str, Any] = read_yaml("labels.yaml")
    return {k: dict(v) for k, v in data.items() if isinstance(v, dict)}


def label(kind: str, code: str | None) -> str:
    if code is None:
        return "—"
    return labels().get(kind, {}).get(str(code), str(code))
