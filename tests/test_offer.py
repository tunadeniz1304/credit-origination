"""Loan offer terms module tests."""
from __future__ import annotations

import pytest

from app.agents.committee_agent import CreditCommitteeAgent
from app.engine.offer import build_offer
from app.engine.scorecard import build_scorecard
from app.models import (
    AggregatedFinancialData,
    Applicant,
    ApplicationStatus,
    LoanApplication,
)


def _approved_decision():
    financial = AggregatedFinancialData(
        identity_no="12345678901", kbb_score=1450, risk_class="DÜŞÜK RİSK",
        total_debt=134070.0, monthly_income=300_000.0,
        employer="Anadolu Bilişim Ltd.", employment_years=8, employment_verified=True,
    )
    application = LoanApplication(
        applicant=Applicant(name="Test", identity_no="12345678901",
                            monthly_income=300_000.0),
        requested_amount=100_000.0, requested_term_months=36,
    )
    return CreditCommitteeAgent().decide(application, financial)


def test_build_offer_for_approved_decision():
    decision = _approved_decision()
    assert decision.status == ApplicationStatus.APPROVED
    scorecard = build_scorecard("APP-X", decision.factors)
    offer = build_offer("APP-X", decision, scorecard)
    assert offer.status == ApplicationStatus.APPROVED
    assert offer.proposed_amount == 1_800_000.0
    assert offer.proposed_term_months == 36
    assert offer.instalment > 0
    assert offer.total_payment > offer.proposed_amount
    assert offer.risk_grade == "A"
    assert offer.rationale_headline


def test_build_offer_rejects_unapproved():
    from app.models import ApplicationStatus as S, CommitteeDecision, CommitteeFactor

    decision = CommitteeDecision(
        status=S.REJECTED, approved=False, factors=[],
        suggested_amount=0.0, suggested_term_months=0,
    )
    with pytest.raises(ValueError):
        build_offer("APP-X", decision)
