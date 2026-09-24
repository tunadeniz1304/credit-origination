"""Domain models shared across the credit pipeline agents (Pydantic v2).

These types are the single source of truth for every stage of the
pipeline: document control, RAG analysis, external API integration,
the credit decision engine and the final reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ApplicationStatus(str, Enum):
    """Lifecycle of a credit application as tracked by the pipeline + queue."""

    QUEUED = "QUEUED"  # accepted and enqueued for processing
    PROCESSING = "PROCESSING"  # a worker is running it
    DOCUMENTS_PENDING = "DOCUMENTS_PENDING"  # documents incomplete; waiting on applicant
    READY = "READY"  # documents complete, awaiting committee review
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"  # processing raised an unrecoverable error


class DocumentRequirement(BaseModel):
    """One mandatory document from the policy, with a Turkish description."""

    model_config = ConfigDict(frozen=True)

    code: str
    description: str


class Applicant(BaseModel):
    """Natural person applying for credit."""

    name: str
    identity_no: str = Field(pattern=r"^\d{11}$")
    monthly_income: float = Field(gt=0)
    submitted_documents: list[str] = Field(default_factory=list)


class LoanApplication(BaseModel):
    """Application as submitted by the applicant."""

    applicant: Applicant
    requested_amount: float = Field(gt=0)
    requested_term_months: int = Field(gt=0, le=600)
    currency: str = "TRY"


class DocumentCheckResult(BaseModel):
    """Outcome of the document-control step."""

    present: list[str] = Field(default_factory=list)  # requirement codes delivered
    missing: list[DocumentRequirement] = Field(default_factory=list)  # not yet delivered
    complete: bool
    request_draft: str = ""  # Turkish request template; empty when complete


class KBBReport(BaseModel):
    """Credit-bureau-style report derived deterministically from an identity no."""

    model_config = ConfigDict(frozen=True)

    score: int  # 0..1900 mock KKB scale
    risk_class: str  # Turkish risk class label
    total_debt: float  # TRY


class EmploymentRecord(BaseModel):
    """e-Devlet-style employment record."""

    model_config = ConfigDict(frozen=True)

    employer: str
    years: int
    verified: bool


class AggregatedFinancialData(BaseModel):
    """Financial picture assembled by the API integration agent."""

    identity_no: str
    kbb_score: int
    risk_class: str
    total_debt: float
    monthly_income: float
    employer: str
    employment_years: int
    employment_verified: bool


class CommitteeFactor(BaseModel):
    """One threshold check evaluated by the credit committee."""

    name: str  # Turkish factor name
    value: float  # observed value
    threshold: float  # allowed bound
    operator: str  # "<=" or ">="
    passed: bool
    unit: str = ""  # displayed unit: "", " puan", "x", " ay"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> str:
        return (
            f"{self.name}: {self.value:,.2f}{self.unit} "
            f"(izin verilen {self.operator} {self.threshold:g}{self.unit})"
        )


class CommitteeDecision(BaseModel):
    """Structured committee verdict + Turkish rationale report."""

    status: ApplicationStatus
    approved: bool
    factors: list[CommitteeFactor] = Field(default_factory=list)
    rationale: str = ""  # Turkish committee summary (cited)
    applicant_letter: str = ""  # plain-Turkish letter to the applicant
    llm_mode: str = "demo"  # live | demo | fallback
    llm_error_kind: str | None = None
    debt_service_ratio: float | None = None
    suggested_amount: float  # approved amount (0 when rejected)
    suggested_term_months: int
    counterfactual_amount: float | None = None  # only for rejections


class PipelineResult(BaseModel):
    """Outcome of running one application through the pipeline."""

    application: LoanApplication
    status: ApplicationStatus
    document_check: DocumentCheckResult
    decision: CommitteeDecision | None = None
    rag_summary: str = ""
    report_json_path: str = ""
    report_pdf_path: str = ""


class ApplicationRecord(BaseModel):
    """API-facing record of a submitted application (queue-aware)."""

    application_id: str
    application: LoanApplication
    status: ApplicationStatus = ApplicationStatus.QUEUED
    task_id: str | None = None
    queue_backend: str = "unknown"
    result: PipelineResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready projection for API responses."""
        return self.model_dump(mode="json")
