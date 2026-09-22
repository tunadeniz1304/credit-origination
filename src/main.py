"""End-to-end credit pipeline runner.

Run with ``python -m src.main``: it exercises three deterministic demo
scenarios — complete documents (approve), missing documents (document
request), and a high-risk applicant (reject).

The reusable :func:`run_application` pipeline is also imported by the test
flow (``tests/test_flow.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agents.api_agent import ApiIntegrationAgent
from src.agents.committee_agent import CreditCommitteeAgent
from src.agents.document_agent import DocumentControlAgent
from src.logger import get_logger
from src.models import (
    Applicant,
    ApplicationStatus,
    CommitteeDecision,
    DocumentCheckResult,
    LoanApplication,
)

# Deterministic documentation of the identities the demo relies on.
# Derived values are produced by src/integrations/mock_providers.py:
#   12345678901 -> KBB 1450, debt 134070 (passes min_kbb_score)
#   34567890123 -> KBB  619, debt 176116 (fails  min_kbb_score)
_ALL_DOCS = ["IDENTITY", "INCOME", "EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"]


@dataclass
class PipelineResult:
    """Outcome of running one application through the pipeline."""

    status: ApplicationStatus
    document_check: DocumentCheckResult
    decision: Optional[CommitteeDecision]


def _applicant(name: str, identity_no: str, income: float, docs: list[str]) -> Applicant:
    return Applicant(name=name, identity_no=identity_no, monthly_income=income, submitted_documents=docs)


def build_scenarios() -> list[LoanApplication]:
    """The three deterministic demo applications."""
    return [
        LoanApplication(
            applicant=_applicant("Ali Yılmaz", "12345678901", 30_000.0, list(_ALL_DOCS)),
            requested_amount=120_000.0,
            requested_term_months=36,
        ),
        LoanApplication(
            applicant=_applicant("Ayşe Demir", "23456789012", 32_000.0, ["IDENTITY", "INCOME"]),
            requested_amount=120_000.0,
            requested_term_months=36,
        ),
        LoanApplication(
            applicant=_applicant("Mehmet Kaya", "34567890123", 8_000.0, list(_ALL_DOCS)),
            requested_amount=200_000.0,
            requested_term_months=72,
        ),
    ]


def run_application(application: LoanApplication) -> PipelineResult:
    """Run the full pipeline: document check -> API collect -> committee."""
    document_agent = DocumentControlAgent()
    check = document_agent.check(application)
    if not check.complete:
        return PipelineResult(status=ApplicationStatus.DOCUMENTS_PENDING, document_check=check, decision=None)

    api_agent = ApiIntegrationAgent()
    financial = api_agent.collect(application.applicant)

    committee = CreditCommitteeAgent()
    decision = committee.decide(application, financial)
    return PipelineResult(status=decision.status, document_check=check, decision=decision)


def main() -> None:
    logger = get_logger("main")
    logger.info("Kredi operasyon pipeline başlatıldı (%d senaryo)", len(build_scenarios()))
    for index, application in enumerate(build_scenarios(), start=1):
        print(f"\n{'=' * 60}\nSENARYO {index}: {application.applicant.name}\n{'=' * 60}")
        result = run_application(application)

        if result.status is ApplicationStatus.DOCUMENTS_PENDING:
            print(f"[{result.status.value}] Belgeler eksik — talep taslağı:\n")
            print(result.document_check.request_draft)
        else:
            print(result.decision.rationale)
            print(f"\nNihai Karar: {result.status.value}")
    logger.info("Kredi operasyon pipeline tamamlandı")


if __name__ == "__main__":
    main()
