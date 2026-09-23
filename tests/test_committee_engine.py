"""Credit committee engine + report generation tests."""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.committee_agent import CreditCommitteeAgent
from app.core.config import Settings
from app.engine.reports import generate_report_files, pdf_to_text
from app.models import (
    AggregatedFinancialData,
    Applicant,
    ApplicationStatus,
    CommitteeDecision,
    DocumentCheckResult,
    LoanApplication,
    PipelineResult,
)

ALL_DOCS = ["IDENTITY", "INCOME", "EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"]
MOCK_PROVIDER_FINANCIALS = {
    "12345678901": dict(kbb_score=1450, risk_class="DÜŞÜK RİSK", total_debt=134070.0,
                        monthly_income=300_000.0),
    "34567890123": dict(kbb_score=619, risk_class="YÜKSEK RİSK", total_debt=176_116.0,
                        monthly_income=30_000.0),
}


def _financial(identity: str, monthly_income: float | None = None) -> AggregatedFinancialData:
    base = MOCK_PROVIDER_FINANCIALS[identity]
    return AggregatedFinancialData(
        identity_no=identity,
        kbb_score=base["kbb_score"],
        risk_class=base["risk_class"],
        total_debt=base["total_debt"],
        monthly_income=monthly_income or base["monthly_income"],
        employer="Anadolu Bilişim Ltd.",
        employment_years=8,
        employment_verified=True,
    )


def _application(monthly_income: float, amount: float, term: int,
                 identity: str = "12345678901") -> LoanApplication:
    return LoanApplication(
        applicant=Applicant(name="Test User", identity_no=identity,
                            monthly_income=monthly_income,
                            submitted_documents=list(ALL_DOCS)),
        requested_amount=amount,
        requested_term_months=term,
    )


def test_committee_approves_when_all_factors_pass():
    decision = CreditCommitteeAgent().decide(
        _application(300_000, 100_000, 36), _financial("12345678901")
    )
    assert isinstance(decision, CommitteeDecision)
    assert decision.status == ApplicationStatus.APPROVED
    assert decision.approved is True
    assert len(decision.factors) == 4
    assert all(factor.passed for factor in decision.factors)
    assert decision.suggested_amount == 300_000 * 6
    assert decision.suggested_term_months == 36
    assert "Kurul Kararı" in decision.rationale


def test_committee_rejects_and_flags_primary_failing_factor():
    decision = CreditCommitteeAgent().decide(
        _application(30_000, 50_000, 24), _financial("34567890123")
    )
    assert decision.status == ApplicationStatus.REJECTED
    assert decision.approved is False
    kbb = next(f for f in decision.factors if f.name == "Kredi Skoru (KKB)")
    assert kbb.passed is False
    assert kbb.value == 619
    assert kbb.operator == ">="
    dti = next(f for f in decision.factors if f.name == "Borç/Gelir Oranı")
    assert dti.passed is False  # 176116 / 30000 = 5.87 > 0.5
    assert decision.rationale


def test_committee_term_over_max_rejects():
    decision = CreditCommitteeAgent().decide(
        _application(100_000, 40_000, 72), _financial("12345678901")
    )
    assert decision.status == ApplicationStatus.REJECTED
    term = next(f for f in decision.factors if f.name == "Vade")
    assert term.passed is False
    assert decision.suggested_term_months == 60
    assert decision.suggested_amount == 300_000 * 6


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        report_output_dir=str(tmp_path / "reports"),
        result_store_dir=str(tmp_path / "results"),
        upload_dir=str(tmp_path / "uploads"),
    )


def _approved_payload() -> PipelineResult:
    decision = CreditCommitteeAgent().decide(
        _application(300_000, 100_000, 36), _financial("12345678901")
    )
    return PipelineResult(
        application=_application(300_000, 100_000, 36),
        status=decision.status,
        document_check=DocumentCheckResult(present=ALL_DOCS, missing=[], complete=True),
        decision=decision,
    )


def test_generate_report_files_writes_json_and_pdf(tmp_path):
    json_path, pdf_path = generate_report_files(
        _approved_payload(), "APP-TEST0001", _settings(tmp_path)
    )
    assert Path(json_path).is_file()
    assert Path(pdf_path).is_file()
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert data["status"] == "APPROVED"
    assert data["decision"]["suggested_amount"] == 1_800_000.0
    assert data["decision"]["factors"][0]["name"] == "Kredi Skoru (KKB)"
    pdf_text = pdf_to_text(pdf_path)
    assert "RAPORU" in pdf_text  # Turkish diacritics are not pypdf-extractable
    assert "KARAR: APPROVED" in pdf_text
    assert "UYGUN" in pdf_text
    assert "Geri Ödeme Plan" in pdf_text
    assert "Toplam Ödeme" in pdf_text
    assert "Kompozit Risk Skoru" in pdf_text
    assert "100/100" in pdf_text


def test_pipeline_runs_end_to_end_and_writes_reports(tmp_path):
    import asyncio

    from app.engine.pipeline import ApplicationPipeline

    settings = _settings(tmp_path)
    with asyncio.Runner() as runner:
        result = runner.run(
            ApplicationPipeline(settings).run(
                _application(300_000, 100_000, 36), application_id="APP-TEST0001"
            )
        )
    assert result.status == ApplicationStatus.APPROVED
    assert result.decision is not None
    assert result.document_check.complete is True
    assert result.rag_summary == ""
    assert result.report_json_path and result.report_pdf_path
    assert Path(result.report_json_path).is_file()
    assert Path(result.report_pdf_path).is_file()
