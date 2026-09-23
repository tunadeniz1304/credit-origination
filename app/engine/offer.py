"""Loan offer terms: combine decision + scorecard + repayment plan.

Deterministic offer generator producing a compact, BDDK-style offer summary:
the committee's suggested amount and term, the composite risk grade from the
scorecard and the first-instalment details from the amortization schedule.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.engine.schedule import build_schedule
from app.engine.scorecard import RiskScorecard
from app.models import ApplicationStatus, CommitteeDecision


class LoanOffer(BaseModel):
    """One deterministically priced credit offer."""

    application_id: str
    status: ApplicationStatus
    proposed_amount: float
    proposed_term_months: int
    annual_rate: float
    instalment: float
    total_payment: float
    total_interest: float
    risk_grade: str
    rationale_headline: str = ""


def build_offer(
    application_id: str,
    decision: CommitteeDecision,
    scorecard: RiskScorecard | None = None,
) -> LoanOffer:
    """Assemble the offer for an approved decision (raises otherwise)."""
    if decision.status != ApplicationStatus.APPROVED:
        raise ValueError("offer can only be built for approved decisions")
    schedule = build_schedule(
        application_id,
        principal=decision.suggested_amount,
        term_months=decision.suggested_term_months,
    )
    grade = scorecard.grade if scorecard else "n/a"
    headline = (
        "Onaylanan kredi: uygun koşullarla kullandırılabilir."
        if grade in ("A", "B")
        else "Onaylanan kredi: temkinli koşullarla kullandırılabilir."
    )
    return LoanOffer(
        application_id=application_id,
        status=decision.status,
        proposed_amount=decision.suggested_amount,
        proposed_term_months=decision.suggested_term_months,
        annual_rate=schedule.annual_rate,
        instalment=schedule.instalment,
        total_payment=schedule.total_payment,
        total_interest=schedule.total_interest,
        risk_grade=grade,
        rationale_headline=headline,
    )
