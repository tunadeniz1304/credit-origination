"""Domain models shared across the credit pipeline agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ApplicationStatus(str, Enum):
    """Lifecycle of a credit application as tracked by the pipeline."""

    DOCUMENTS_PENDING = "DOCUMENTS_PENDING"  # documents incomplete; waiting on applicant
    READY = "READY"  # documents complete, awaiting committee review
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class DocumentRequirement:
    """One mandatory document from the policy, with a Turkish description."""

    code: str
    description: str


@dataclass
class Applicant:
    """Natural person applying for credit."""

    name: str
    identity_no: str
    monthly_income: float
    submitted_documents: list[str] = field(default_factory=list)


@dataclass
class LoanApplication:
    applicant: Applicant
    requested_amount: float
    requested_term_months: int
    currency: str = "TRY"


@dataclass
class DocumentCheckResult:
    """Outcome of the document-control step."""

    present: list[str]  # requirement codes delivered by the applicant
    missing: list[DocumentRequirement]  # requirement codes not yet delivered
    complete: bool
    request_draft: str  # Turkish request template; empty when complete


@dataclass(frozen=True)
class KBBReport:
    """Credit-bureau-style report derived deterministically from an identity no."""

    score: int  # 0..1900 mock KKB scale
    risk_class: str  # Turkish risk class label
    total_debt: float  # TRY


@dataclass(frozen=True)
class EmploymentRecord:
    """e-Devlet-style employment record."""

    employer: str
    years: int
    verified: bool


@dataclass
class AggregatedFinancialData:
    """Financial picture assembled by the API integration agent."""

    identity_no: str
    kbb_score: int
    risk_class: str
    total_debt: float
    monthly_income: float
    employer: str
    employment_years: int
    employment_verified: bool


@dataclass(frozen=True)
class CommitteeFactor:
    """One threshold check evaluated by the credit committee."""

    name: str  # Turkish factor name
    value: float  # observed value
    threshold: float  # allowed bound
    operator: str  # "<=" or ">="
    passed: bool
    unit: str = ""  # displayed unit: "", " puan", "x", " ay"

    @property
    def summary(self) -> str:
        return (
            f"{self.name}: {self.value:,.2f}{self.unit} "
            f"(izin verilen {self.operator} {self.threshold:g}{self.unit})"
        )


@dataclass
class CommitteeDecision:
    """Structured committee verdict + Turkish rationale report."""

    status: ApplicationStatus
    approved: bool
    factors: list[CommitteeFactor]
    rationale: str  # Turkish committee report
    suggested_amount: float
    suggested_term_months: int
