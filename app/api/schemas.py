"""Pydantic request/response schemas for the application API."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

TC_KIMLIK_PATTERN = r"^\d{11}$"
OFFERED_CURRENCIES = Literal["TRY"]
ApplicantName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]


class ApplicationSubmit(BaseModel):
    """Applicant payload accepted by ``POST /api/v1/applications``."""

    name: ApplicantName
    identity_no: str = Field(pattern=TC_KIMLIK_PATTERN)
    monthly_income: float = Field(gt=0)
    requested_amount: float = Field(gt=0)
    requested_term_months: int = Field(gt=0, le=600)
    currency: OFFERED_CURRENCIES = "TRY"
    submitted_documents: list[str] = Field(default_factory=list)
