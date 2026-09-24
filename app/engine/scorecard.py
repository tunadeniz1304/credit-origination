"""Deterministic BDDK-style composite risk score from committee factors.

Each committee factor (credit score, debt-to-income ratio, loan-to-income
multiplier and term) is mapped to a fixed weight. Passing factors score the
full 100 points; failing factors score proportionally to how close the observed
value stays to the allowed threshold. The weighted contributions are summed
into a single 0..100 score that is graded A..E and is fully reproducible —
there is no randomness anywhere in the computation.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.models import CommitteeFactor


class ScorecardRow(BaseModel):
    """One factor's contribution to the composite risk score."""

    name: str
    weight: float
    score: float
    max_score: float
    contribution: float


class RiskScorecard(BaseModel):
    """Composite risk scorecard with per-factor rows and an overall grade."""

    application_id: str
    total_score: float
    grade: str
    rows: list[ScorecardRow]


# Fixed BDDK-style weights keyed by the Turkish committee factor names.
FACTOR_WEIGHTS: dict[str, float] = {
    "Kredi Skoru (KKB)": 0.40,
    "Borç/Gelir Oranı": 0.30,
    "Kredi/Gelir Çarpanı": 0.20,
    "Vade": 0.10,
}


def _grade(total_score: float) -> str:
    """Map a 0..100 total score to a letter grade."""
    if total_score >= 80:
        return "A"
    if total_score >= 65:
        return "B"
    if total_score >= 50:
        return "C"
    if total_score >= 35:
        return "D"
    return "E"


def _row_score(factor: CommitteeFactor | object) -> float:
    """Per-row score in 0..100; 100 when the factor passes."""
    if factor.passed:
        return 100.0
    distance = abs(factor.value - factor.threshold)
    denominator = max(abs(factor.threshold), 1.0)
    return max(0.0, min(100.0, 100.0 * (1.0 - distance / denominator)))


def build_scorecard(application_id: str, factors: object) -> RiskScorecard:
    """Build a weighted composite scorecard from a sequence of committee factors.

    ``factors`` accepts any iterable of objects exposing ``name``, ``passed``,
    ``value`` and ``threshold`` (duck-typed); ``CommitteeFactor`` is used only
    for the type hint. Unknown factor names receive a zero weight and zero
    score, so they never influence the total.
    """
    rows: list[ScorecardRow] = []
    for factor in factors:  # type: ignore[union-attr]
        weight = FACTOR_WEIGHTS.get(factor.name, 0.0)
        score = _row_score(factor) if weight > 0 else 0.0
        rows.append(
            ScorecardRow(
                name=factor.name,
                weight=weight,
                score=score,
                max_score=100.0,
                contribution=round(score * weight, 2),
            )
        )

    total_score = round(sum(row.contribution for row in rows), 2)
    return RiskScorecard(
        application_id=application_id,
        total_score=total_score,
        grade=_grade(total_score),
        rows=rows,
    )
