"""Scorecard module tests."""

from __future__ import annotations

from app.engine.scorecard import build_scorecard
from app.models import CommitteeFactor


def _factor(
    name: str, value: float, threshold: float, operator: str, passed: bool
) -> CommitteeFactor:
    return CommitteeFactor(
        name=name,
        value=value,
        threshold=threshold,
        operator=operator,
        passed=passed,
    )


def _all_pass() -> list[CommitteeFactor]:
    return [
        _factor("Kredi Skoru (KKB)", 1200, 1100, ">=", True),
        _factor("Borç Servis Oranı", 0.2, 0.6, "<=", True),
        _factor("Kredi/Gelir Çarpanı", 3.0, 6.0, "<=", True),
        _factor("Vade", 24, 60, "<=", True),
    ]


def _all_fail() -> list[CommitteeFactor]:
    return [
        _factor("Kredi Skoru (KKB)", 0, 1100, ">=", False),
        _factor("Borç Servis Oranı", 2.0, 0.6, "<=", False),
        _factor("Kredi/Gelir Çarpanı", 50.0, 6.0, "<=", False),
        _factor("Vade", 500, 60, "<=", False),
    ]


def test_all_factors_pass_yields_full_score_and_grade_a():
    card = build_scorecard("app-1", _all_pass())

    assert card.total_score == 100.0
    assert card.grade == "A"
    assert card.application_id == "app-1"
    assert sum(row.contribution for row in card.rows) == card.total_score
    assert all(row.score == 100.0 for row in card.rows)


def test_failing_kbb_factor_lowers_score_and_grade():
    factors = _all_pass()
    factors[0] = _factor("Kredi Skoru (KKB)", 500, 1100, ">=", False)
    card = build_scorecard("app-2", factors)

    kbb_row = card.rows[0]
    assert kbb_row.score < 100.0
    assert kbb_row.weight == 0.40
    assert card.total_score < 100.0
    assert card.grade != "A"
    assert card.total_score == sum(row.contribution for row in card.rows)


def test_full_fail_scores_lower_than_full_approve():
    approve = build_scorecard("app-3", _all_pass())
    fail = build_scorecard("app-4", _all_fail())

    assert approve.total_score == 100.0
    assert fail.total_score < approve.total_score
