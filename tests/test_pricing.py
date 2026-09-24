"""Risk-based pricing tests: RAROC solve, taxes, legal cap, APR, schedule."""

from __future__ import annotations

import pytest

from app.core.rules import load_pricing
from app.pricing.engine import irb_capital, minimum_rate, quote, raroc


def test_minimum_rate_meets_target_raroc():
    cfg = load_pricing()
    rate = minimum_rate(0.03, "IHTIYAC", cfg)
    assert raroc(rate, 0.03, "IHTIYAC", cfg) == pytest.approx(
        cfg.product("IHTIYAC").target_raroc, abs=1e-4
    )


def test_rate_increases_with_pd():
    rates = [
        quote(pd=pd, amount=100_000, term_months=24).annual_rate for pd in (0.01, 0.05, 0.1, 0.2)
    ]
    assert rates == sorted(rates)


def test_capital_increases_with_pd_until_high_levels():
    assert irb_capital(0.01, 0.55, 0.15, 0.999) < irb_capital(0.10, 0.55, 0.15, 0.999)


def test_schedule_amortises_with_taxes_on_interest():
    q = quote(pd=0.02, amount=120_000, term_months=24)
    assert len(q.schedule) == 24
    assert q.schedule[-1].closing_balance == 0.0
    first = q.schedule[0]
    assert first.kkdf == pytest.approx(first.interest * 0.15, abs=0.02)
    assert first.bsmv == pytest.approx(first.interest * 0.15, abs=0.02)
    assert sum(r.principal for r in q.schedule) == pytest.approx(120_000, abs=0.05)
    assert q.gross_monthly_rate == pytest.approx(q.annual_rate / 12 * 1.30, rel=1e-4)


def test_apr_exceeds_contract_rate_because_of_taxes_and_fee():
    q = quote(pd=0.02, amount=120_000, term_months=24)
    assert q.apr > q.annual_rate
    assert q.upfront_fee == pytest.approx(120_000 * 0.005 * 1.15, abs=0.01)


def test_legal_cap_flag():
    q = quote(pd=0.02, amount=50_000, term_months=12, rate_override=0.9)
    assert q.exceeds_cap
    assert not quote(pd=0.02, amount=50_000, term_months=12).exceeds_cap


def test_quote_without_schedule_is_light():
    assert quote(pd=0.05, amount=10_000, term_months=6, include_schedule=False).schedule == []
