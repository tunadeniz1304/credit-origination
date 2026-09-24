"""Citation and number guard for LLM narratives.

Every number and claim in a credit memorandum must be traceable to a field of
the decision context (``field_id`` such as ``f:dsr``). The guard rejects text
that cites unknown fields or states numbers that do not match any cited fact
(within tolerance), which triggers a repair attempt and then the deterministic
fallback. Turkish number formats (``180.000``, ``0,58``, ``%58``) are handled.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[(f:[A-Za-z0-9_.]+)\]")
_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")
# Legal references, the 12-month PD horizon and calendar constants that may
# appear without a citation.
ALLOWED_CONSTANTS: frozenset[float] = frozenset({6698.0, 5411.0, 11.0, 12.0, 30.0, 173.0, 2020.0})
SMALL_INT_LIMIT = 10


@dataclass
class GuardResult:
    """Outcome of validating one narrative."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    cited: set[str] = field(default_factory=set)


def _candidates(token: str) -> set[float]:
    """All plausible numeric readings of a token (TR and EN conventions)."""
    readings: set[float] = set()
    tr = token.replace(".", "").replace(",", ".")
    en = token.replace(",", "")
    for raw in (tr, en):
        try:
            readings.add(float(raw))
        except ValueError:
            continue
    return readings


def _embedded(text: str, start: int, end: int) -> bool:
    """True when the digits belong to an identifier such as ``APP-9F3A`` or ``R01``."""
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    if before.isalpha() or after.isalpha() or after == "_":
        return True
    return before in "_-" and start > 1 and text[start - 2].isalnum()


def extract_numbers(text: str) -> list[tuple[str, set[float]]]:
    """Numbers mentioned in ``text`` (citation markers and identifiers excluded)."""
    stripped = CITATION_RE.sub(" ", text)
    return [
        (m.group(0), _candidates(m.group(0)))
        for m in _NUMBER_RE.finditer(stripped)
        if not _embedded(stripped, m.start(), m.end())
    ]


def _matches(value: float, fact: float) -> bool:
    for target in (fact, fact * 100.0):
        tolerance = max(abs(target) * 0.01, 0.5 if abs(target) >= 1 else 0.005)
        if abs(value - target) <= tolerance:
            return True
    return False


def _numeric_facts(facts: Mapping[str, object], ids: Iterable[str]) -> list[float]:
    values: list[float] = []
    for fid in ids:
        value = facts.get(fid)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _ignorable(readings: set[float]) -> bool:
    return any(
        (r.is_integer() and 0 <= r <= SMALL_INT_LIMIT) or r in ALLOWED_CONSTANTS for r in readings
    )


def check_numbers(text: str, allowed: list[float]) -> list[str]:
    """Return an error for each number not backed by an allowed value."""
    errors: list[str] = []
    for token, readings in extract_numbers(text):
        if not readings or _ignorable(readings):
            continue
        if not any(_matches(r, fact) for r in readings for fact in allowed):
            errors.append(f"unsupported number '{token}'")
    return errors


def validate_cited_text(
    text: str,
    facts: Mapping[str, object],
    *,
    require_citation: bool = True,
) -> GuardResult:
    """Validate a narrative whose claims carry inline ``[f:...]`` citations."""
    cited = set(CITATION_RE.findall(text))
    errors = [f"unknown field '{fid}'" for fid in sorted(cited) if fid not in facts]
    if require_citation and not cited:
        errors.append("no citations present")
    errors += check_numbers(text, _numeric_facts(facts, cited))
    return GuardResult(ok=not errors, errors=errors, cited=cited)


def validate_claims(
    claims: Iterable[tuple[str, list[str]]],
    facts: Mapping[str, object],
) -> GuardResult:
    """Validate structured claims ``(text, field_ids)`` claim by claim."""
    errors: list[str] = []
    cited: set[str] = set()
    for text, field_ids in claims:
        unknown = [fid for fid in field_ids if fid not in facts]
        errors += [f"unknown field '{fid}'" for fid in unknown]
        cited.update(field_ids)
        errors += check_numbers(text, _numeric_facts(facts, field_ids))
    return GuardResult(ok=not errors, errors=errors, cited=cited)


def validate_plain_text(text: str, facts: Mapping[str, object]) -> GuardResult:
    """Validate an applicant-facing text: numbers must exist anywhere in the facts."""
    errors = check_numbers(text, _numeric_facts(facts, facts.keys()))
    return GuardResult(ok=not errors, errors=errors)


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[f:[A-Za-z0-9_.]+\]", "", text)
