"""API request schemas (pydantic v2) — the single source of truth for inputs."""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, EmailStr, Field, StringConstraints, field_validator

from app.kyc.tckn import is_valid_tckn

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=120)]
PHONE_RE = re.compile(r"^(?:\+90|0)?5\d{9}$")
IBAN_RE = re.compile(r"^TR\d{24}$")


class ConsentSet(BaseModel):
    kvkk_aydinlatma: bool = Field(description="KVKK aydınlatma metni okundu")
    acik_riza: bool = Field(description="Açık rıza (otomatik karar ve profilleme)")
    kkb_sorgu: bool = Field(description="KKB/Findeks sorgu izni")
    edevlet_sorgu: bool = True
    acik_bankacilik: bool = False


class ApplicationCreate(BaseModel):
    """Applicant payload accepted by ``POST /api/v1/applications``."""

    name: Name
    identity_no: str = Field(pattern=r"^\d{11}$")
    birth_date: date | None = None
    phone: str | None = None
    email: EmailStr | None = None
    address: str | None = Field(default=None, max_length=300)
    iban: str | None = None
    gender: Literal["K", "E"] | None = Field(
        default=None, description="Yalnızca adillik izlemesi için"
    )
    province: str | None = Field(default=None, max_length=40)
    monthly_income: float = Field(gt=0, le=10_000_000)
    employment_type: Literal["MAASLI", "SERBEST", "EMEKLI"] = "MAASLI"
    employer_name: str | None = Field(default=None, max_length=120)
    product: Literal["IHTIYAC", "TASIT"] = "IHTIYAC"
    requested_amount: float = Field(gt=0, le=10_000_000)
    requested_term_months: int = Field(gt=0, le=120)
    currency: Literal["TRY"] = "TRY"
    device_id: str | None = Field(default=None, max_length=64)
    consents: ConsentSet

    @field_validator("identity_no")
    @classmethod
    def _tckn(cls, value: str) -> str:
        if not is_valid_tckn(value):
            raise ValueError("geçersiz T.C. kimlik numarası (checksum)")
        return value

    @field_validator("phone")
    @classmethod
    def _phone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        compact = re.sub(r"[\s()-]", "", value)
        if not PHONE_RE.match(compact):
            raise ValueError("geçersiz cep telefonu")
        return compact

    @field_validator("iban")
    @classmethod
    def _iban(cls, value: str | None) -> str | None:
        if value is None:
            return value
        compact = value.replace(" ", "").upper()
        if not IBAN_RE.match(compact):
            raise ValueError("geçersiz TR IBAN")
        return compact

    @field_validator("consents")
    @classmethod
    def _mandatory_consents(cls, value: ConsentSet) -> ConsentSet:
        if not (value.kvkk_aydinlatma and value.acik_riza and value.kkb_sorgu):
            raise ValueError("KVKK aydınlatma, açık rıza ve KKB sorgu izni zorunludur")
        return value


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class RegisterRequest(LoginRequest):
    full_name: Name


class OpenBankingConsentRequest(BaseModel):
    scopes: list[str] = Field(
        default_factory=lambda: ["hesap_bilgisi", "hesap_hareketleri", "bakiye"]
    )
    days: int = Field(default=90, ge=1, le=180)


class ObjectionRequest(BaseModel):
    reason: str = Field(min_length=10, max_length=2000)


class QuoteRequest(BaseModel):
    amount: float = Field(gt=0, le=10_000_000)
    term_months: int = Field(gt=0, le=120)
    product: Literal["IHTIYAC", "TASIT"] = "IHTIYAC"
    pd: float | None = Field(default=None, gt=0, lt=1)
    risk_band: Literal["A", "B", "C", "D"] | None = None


class ReviewDecisionRequest(BaseModel):
    action: Literal["ONAY", "RET", "BELGE_ISTE"]
    justification: str = Field(min_length=15, max_length=4000, description="Zorunlu gerekçe")
    amount: float | None = Field(default=None, gt=0)
    term_months: int | None = Field(default=None, gt=0, le=120)
    annual_rate: float | None = Field(default=None, gt=0, lt=2)


class CheckerRequest(BaseModel):
    approve: bool
    note: str = Field(min_length=5, max_length=2000)


class FieldCorrection(BaseModel):
    value: str = Field(min_length=1, max_length=300)
    note: str = Field(default="", max_length=500)


class ObjectionResolution(BaseModel):
    upheld: bool
    note: str = Field(min_length=10, max_length=4000)


class PolicyQuestion(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    application_id: str | None = None


class RuleSetDraft(BaseModel):
    version: str = Field(pattern=r"^policy_v\d+[a-z0-9_]*$")
    content: str = Field(min_length=20, max_length=200_000)
