"""End-to-end test flow for the credit operations pipeline.

Deterministic (no network): all external data is mocked and derived purely
from the applicant's identity number. Each scenario asserts the observable
contract (status, factors, request draft) rather than internal wiring.
"""
from __future__ import annotations

from src.main import build_scenarios, run_application
from src.models import ApplicationStatus


def test_complete_application_is_approved():
    application = build_scenarios()[0]
    result = run_application(application)

    assert result.status is ApplicationStatus.APPROVED
    assert result.document_check.complete is True
    assert result.document_check.missing == []
    assert result.decision is not None
    assert result.decision.approved is True
    assert all(factor.passed for factor in result.decision.factors)
    assert result.decision.suggested_amount == application.requested_amount
    assert "KKB" in result.decision.rationale


def test_missing_documents_stop_with_request_draft():
    application = build_scenarios()[1]
    result = run_application(application)

    assert result.status is ApplicationStatus.DOCUMENTS_PENDING
    assert result.document_check.complete is False
    assert result.decision is None  # pipeline stops before committee

    missing_codes = {req.code for req in result.document_check.missing}
    assert missing_codes == {"EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"}
    draft = result.document_check.request_draft
    for code in missing_codes:
        assert code in draft
    # Turkish request template mentions the missing documents' descriptions.
    assert "Son 3 Ay Banka Hesap Ekstresi" in draft
    assert application.applicant.name in draft


def test_high_risk_application_is_rejected():
    application = build_scenarios()[2]
    result = run_application(application)

    assert result.status is ApplicationStatus.REJECTED
    assert result.decision is not None
    assert result.decision.approved is False
    assert not all(factor.passed for factor in result.decision.factors)

    failed = [factor for factor in result.decision.factors if not factor.passed]
    # The KBB score (619 < min 1100) is a guaranteed failed factor for this applicant.
    assert any("KKB" in factor.name for factor in failed)
    # Failed factor names surface in the rationale report.
    for factor in failed:
        assert factor.name in result.decision.rationale
    assert result.decision.suggested_amount == 0.0
