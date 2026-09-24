"""Sanctions / PEP screening with Turkish-aware fuzzy matching.

Names are normalised (Turkish casefolding, diacritics folded to ASCII,
whitespace collapsed) and compared with Jaro-Winkler similarity from
``rapidfuzz`` both on the full name and on the token-sorted form. A score at
or above the threshold is a *potential* match: it refers the application to
a specialist, it never auto-declines.
"""

from __future__ import annotations

import re

from pydantic import BaseModel
from rapidfuzz.distance import JaroWinkler

from app.core.rules import load_sanctions

_TR_MAP = str.maketrans(
    {"ç": "c", "ğ": "g", "ı": "i", "İ": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i"}
)
DEFAULT_THRESHOLD = 0.92


def normalise_name(name: str) -> str:
    lowered = name.replace("I", "ı").replace("İ", "i").lower()
    folded = lowered.translate(_TR_MAP)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", folded)).strip()


class ScreeningHit(BaseModel):
    name: str
    list: str
    type: str
    score: float


class ScreeningResult(BaseModel):
    matched: bool
    best_score: float
    hits: list[ScreeningHit]
    list_version: str


def _similarity(a: str, b: str) -> float:
    direct = JaroWinkler.similarity(a, b)
    sorted_a = " ".join(sorted(a.split()))
    sorted_b = " ".join(sorted(b.split()))
    return max(direct, JaroWinkler.similarity(sorted_a, sorted_b))


def screen_name(name: str, threshold: float = DEFAULT_THRESHOLD) -> ScreeningResult:
    watchlist = load_sanctions()
    query = normalise_name(name)
    hits: list[ScreeningHit] = []
    best = 0.0
    for entry in watchlist.entries:
        score = _similarity(query, normalise_name(entry.name))
        best = max(best, score)
        if score >= threshold:
            hits.append(
                ScreeningHit(
                    name=entry.name, list=entry.list, type=entry.type, score=round(score, 4)
                )
            )
    hits.sort(key=lambda h: h.score, reverse=True)
    return ScreeningResult(
        matched=bool(hits), best_score=round(best, 4), hits=hits, list_version=watchlist.version
    )
