"""Deterministic loan amortization (repayment plan) generation.

Standard annuity repayment: equal monthly instalments whose present-value
sum equals the principal at the quoted annual rate. Produces a BDDK-style
repayment plan (vade planı) for the report and API surface.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, Field

DEFAULT_ANNUAL_RATE = 0.30  # mock policy: 30% annual simple rate
MONTHS_PER_YEAR = 12


class RepaymentRow(BaseModel):
    """One monthly instalment of the amortization plan."""

    period: int  # 1-based month index
    principal_balance: float  # outstanding principal at the start of the month
    instalment: float  # fixed monthly payment (annuity)
    interest: float  # interest portion of this instalment
    principal_paid: float  # principal portion of this instalment


class RepaymentSchedule(BaseModel):
    """Full amortization plan for an approved credit."""

    application_id: str
    principal: float
    annual_rate: float
    term_months: int
    monthly_rate: float
    instalment: float
    total_payment: float
    total_interest: float
    rows: list[RepaymentRow] = Field(default_factory=list)


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_schedule(
    application_id: str,
    principal: float,
    term_months: int,
    annual_rate: float = DEFAULT_ANNUAL_RATE,
) -> RepaymentSchedule:
    """Compute the annuity repayment plan for a loan."""
    months = max(1, term_months)
    p = Decimal(str(principal))
    i = Decimal(str(annual_rate)) / MONTHS_PER_YEAR

    if i == 0:
        instalment = p / months
    else:
        factor = (Decimal(1) + i) ** months
        instalment = p * i * factor / (factor - Decimal(1))

    instalment = instalment.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    rows: list[RepaymentRow] = []
    balance = p
    total_interest = Decimal("0")
    for period in range(1, months + 1):
        interest = (balance * i).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        paid = instalment - interest
        if paid > balance:
            paid = balance  # final row: settle the remainder
        balance = (balance - paid).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total_interest += interest
        rows.append(
            RepaymentRow(
                period=period,
                principal_balance=_money(balance + paid),
                instalment=_money(instalment),
                interest=_money(interest),
                principal_paid=_money(paid),
            )
        )

    total_payment = p + total_interest
    return RepaymentSchedule(
        application_id=application_id,
        principal=_money(p),
        annual_rate=annual_rate,
        term_months=months,
        monthly_rate=float(i),
        instalment=_money(instalment),
        total_payment=_money(total_payment),
        total_interest=_money(total_interest),
        rows=rows,
    )
