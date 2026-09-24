"""Deterministic mock provider payloads.

KKB (credit bureau) and e-Devlet (employment) values are derived from a
stable SHA-256 digest of the identity number — never Python's salted
``hash()`` — so every identity maps to the same values across processes,
machines and runs. The sample identities are pinned by the test suite:

    12345678901 -> KBB 1450, debt 134070 (passes min_kbb_score)
    34567890123 -> KBB  619, debt 176116 (fails  min_kbb_score)
"""

from __future__ import annotations

import hashlib

from app.models import EmploymentRecord, KBBReport

EMPLOYERS = (
    "Yıldız Holding A.Ş.",
    "Anadolu Bilişim Ltd.",
    "Ege Lojistik San. Tic.",
    "KARBON Enerji A.Ş.",
    "Bereket Gıda Pazarlama",
)

RISK_CLASSES = (
    (1500, "ÇOK DÜŞÜK RİSK"),
    (1100, "DÜŞÜK RİSK"),
    (700, "ORTA RİSK"),
    (0, "YÜKSEK RİSK"),
)

KBB_MAX_SCORE = 1900
KBB_MIN_SCORE = 300
DEBT_MIN = 5_000.0
DEBT_MAX = 180_000.0


def stable_digest(value: str) -> int:
    """Stable, process-independent 64-bit digest of a string."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def range_value(value: str, lo: int, hi: int) -> int:
    """Deterministic integer in ``[lo, hi)`` per (value, lo, hi)."""
    return lo + stable_digest(f"{value}|{lo}|{hi}") % (hi - lo)


def derive_kbb_report(identity_no: str) -> KBBReport:
    """Deterministic mock credit-bureau report for an identity number."""
    score = range_value(identity_no, KBB_MIN_SCORE, KBB_MAX_SCORE + 1)
    total_debt = range_value(identity_no, int(DEBT_MIN), int(DEBT_MAX) + 1)
    risk_class = next(label for threshold, label in RISK_CLASSES if score >= threshold)
    return KBBReport(score=score, risk_class=risk_class, total_debt=float(total_debt))


def derive_employment_record(identity_no: str) -> EmploymentRecord:
    """Deterministic mock e-Devlet employment record for an identity number."""
    index = stable_digest(identity_no) % len(EMPLOYERS)
    return EmploymentRecord(
        employer=EMPLOYERS[index],
        years=range_value(identity_no, 0, 35),
        verified=range_value(identity_no, 0, 10) != 0,
    )
