"""Amortization schedule engine tests."""
from __future__ import annotations

import pytest

from app.engine.schedule import DEFAULT_ANNUAL_RATE, build_schedule


def test_schedule_lays_out_full_term_and_settles():
    schedule = build_schedule("APP-TEST", principal=1_000_000.0, term_months=3)
    assert schedule.term_months == 3
    assert len(schedule.rows) == 3
    assert schedule.rows[0].principal_balance == pytest.approx(1_000_000.0, abs=0.01)
    assert schedule.total_payment > schedule.principal
    assert schedule.total_interest == pytest.approx(
        schedule.total_payment - schedule.principal, abs=0.2
    )
    # principal is fully repaid across the rows (last row settles the remainder)
    repaid = sum(row.principal_paid for row in schedule.rows)
    assert repaid == pytest.approx(1_000_000.0, abs=0.2)


def test_schedule_instalments_are_constant():
    schedule = build_schedule("APP-TEST", principal=500_000.0, term_months=6)
    instalments = {round(row.instalment, 2) for row in schedule.rows}
    assert len(instalments) <= 2  # constant except for the rounded final row


def test_schedule_total_is_close_to_instalment_times_months():
    schedule = build_schedule("APP-TEST", principal=120_000.0, term_months=12)
    assert schedule.total_payment == pytest.approx(
        schedule.instalment * schedule.term_months, abs=1.0
    )
    assert schedule.monthly_rate == pytest.approx(DEFAULT_ANNUAL_RATE / 12, abs=1e-6)
