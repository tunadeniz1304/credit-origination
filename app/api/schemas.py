"""Pydantic request/response schemas for the application API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicationSubmit(BaseModel):
    """Applicant payload accepted by ``POST /api/v1/applications``."""

    name: str
    identity_no: str
    monthly_income: float = Field(gt=0)
    requested_amount: float = Field(gt=0)
    requested_term_months: int = Field(gt=0, le=600)
    currency: str = "TRY"
    submitted_documents: list[str] = Field(default_factory=list)
