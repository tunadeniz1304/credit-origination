"""SQLAlchemy 2.0 ORM models (PostgreSQL in Docker, SQLite locally/tests).

PII columns hold Fernet ciphertext (``*_enc``) plus HMAC blind indexes
(``*_bidx``) for equality lookups. ``decisions`` stores everything needed to
replay a decision bit-for-bit: rule-set version, model version, the feature
snapshot and its hash. ``audit_log`` is hash-chained; ``outbox`` implements the
transactional outbox pattern.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    """Declarative base with JSON mapping for ``dict``/``list`` columns."""

    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(32), index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Applicant(TimestampMixin, Base):
    __tablename__ = "applicants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    tckn_enc: Mapped[str | None] = mapped_column(Text)
    tckn_bidx: Mapped[str | None] = mapped_column(String(64), index=True)
    name_enc: Mapped[str | None] = mapped_column(Text)
    phone_enc: Mapped[str | None] = mapped_column(Text)
    phone_bidx: Mapped[str | None] = mapped_column(String(64), index=True)
    email_enc: Mapped[str | None] = mapped_column(Text)
    address_enc: Mapped[str | None] = mapped_column(Text)
    address_bidx: Mapped[str | None] = mapped_column(String(64), index=True)
    iban_enc: Mapped[str | None] = mapped_column(Text)
    iban_bidx: Mapped[str | None] = mapped_column(String(64), index=True)
    birth_date: Mapped[date | None] = mapped_column(Date)
    # Protected / proxy attributes: monitoring only, never model features.
    gender: Mapped[str | None] = mapped_column(String(8))
    province: Mapped[str | None] = mapped_column(String(40))
    anonymized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Application(TimestampMixin, Base):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    applicant_id: Mapped[str] = mapped_column(ForeignKey("applicants.id"), index=True)
    owner_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    product: Mapped[str] = mapped_column(String(32), default="IHTIYAC")
    requested_amount: Mapped[float] = mapped_column(Float)
    requested_term_months: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="TRY")
    declared_income: Mapped[float] = mapped_column(Float)
    employment_type: Mapped[str] = mapped_column(String(16), default="MAASLI")
    employer_name: Mapped[str | None] = mapped_column(String(120))
    device_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona: Mapped[str | None] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sla_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    assigned_to: Mapped[str | None] = mapped_column(String(64), index=True)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    kyc: Mapped[dict[str, Any]] = mapped_column(default=dict)
    letters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    reports: Mapped[dict[str, Any]] = mapped_column(default=dict)
    flags: Mapped[list[Any]] = mapped_column(default=list)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    task_id: Mapped[str | None] = mapped_column(String(64))
    queue_backend: Mapped[str | None] = mapped_column(String(16))

    applicant: Mapped[Applicant] = relationship(lazy="joined")
    events: Mapped[list[ApplicationEvent]] = relationship(
        back_populates="application", order_by="ApplicationEvent.id", lazy="selectin"
    )


class ApplicationEvent(Base):
    __tablename__ = "application_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped[Application] = relationship(back_populates="events")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    code: Mapped[str] = mapped_column(String(32))
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    mime: Mapped[str] = mapped_column(String(64))
    size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default="YUKLENDI")
    text_source: Mapped[str | None] = mapped_column(String(16))  # digital | ocr | none
    fraud_score: Mapped[float] = mapped_column(Float, default=0.0)
    fraud_signals: Mapped[list[Any]] = mapped_column(default=list)
    uploaded_by: Mapped[str | None] = mapped_column(String(64))


class ExtractedField(TimestampMixin, Base):
    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    page: Mapped[int | None] = mapped_column(Integer)
    bbox: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="regex")
    corrected_by: Mapped[str | None] = mapped_column(String(64))


class BureauReport(TimestampMixin, Base):
    __tablename__ = "bureau_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    provider: Mapped[str] = mapped_column(String(24))
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)


class CashflowFeatures(TimestampMixin, Base):
    __tablename__ = "cashflow_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    features: Mapped[dict[str, Any]] = mapped_column(default=dict)
    monthly: Mapped[list[Any]] = mapped_column(default=list)
    categories: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Decision(TimestampMixin, Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="engine")  # engine|override|objection
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    conditional: Mapped[bool] = mapped_column(Boolean, default=False)
    pd: Mapped[float | None] = mapped_column(Float)
    score_points: Mapped[float | None] = mapped_column(Float)
    risk_band: Mapped[str | None] = mapped_column(String(4))
    reason_codes: Mapped[list[Any]] = mapped_column(default=list)
    rule_set_version: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    feature_snapshot: Mapped[dict[str, Any]] = mapped_column(default=dict)
    feature_hash: Mapped[str] = mapped_column(String(64), index=True)
    rule_results: Mapped[list[Any]] = mapped_column(default=list)
    limits: Mapped[dict[str, Any]] = mapped_column(default=dict)
    pricing: Mapped[dict[str, Any]] = mapped_column(default=dict)
    counterfactuals: Mapped[list[Any]] = mapped_column(default=list)
    explanation: Mapped[dict[str, Any]] = mapped_column(default=dict)
    challenger: Mapped[dict[str, Any]] = mapped_column(default=dict)
    narratives: Mapped[dict[str, Any]] = mapped_column(default=dict)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    decided_by: Mapped[str] = mapped_column(String(64), default="engine")


class Offer(TimestampMixin, Base):
    __tablename__ = "offers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("decisions.id"))
    amount: Mapped[float] = mapped_column(Float)
    term_months: Mapped[int] = mapped_column(Integer)
    annual_rate: Mapped[float] = mapped_column(Float)
    instalment: Mapped[float] = mapped_column(Float)
    apr: Mapped[float] = mapped_column(Float)
    total_payment: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="SUNULDU")
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Review(TimestampMixin, Base):
    """Specialist review / override with maker-checker support."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    maker: Mapped[str] = mapped_column(String(64))
    maker_role: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(16))  # ONAY | RET | BELGE_ISTE
    amount: Mapped[float | None] = mapped_column(Float)
    term_months: Mapped[int | None] = mapped_column(Integer)
    annual_rate: Mapped[float | None] = mapped_column(Float)
    justification: Mapped[str] = mapped_column(Text)
    is_override: Mapped[bool] = mapped_column(Boolean, default=False)
    required_role: Mapped[str] = mapped_column(String(32))
    four_eyes: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="ONAY_BEKLIYOR")
    checker: Mapped[str | None] = mapped_column(String(64))
    checker_note: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Objection(TimestampMixin, Base):
    """KVKK art. 11 objection to an automated adverse decision."""

    __tablename__ = "objections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="ACIK")
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(String(64))
    resolution: Mapped[str | None] = mapped_column(Text)
    letter: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Consent(TimestampMixin, Base):
    __tablename__ = "consents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    type: Mapped[str] = mapped_column(String(32))
    scope: Mapped[list[Any]] = mapped_column(default=list)
    text_version: Mapped[str] = mapped_column(String(16), default="v1")
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelRecord(TimestampMixin, Base):
    """Model inventory entry (SR 26-2 style)."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(32))
    version: Mapped[str] = mapped_column(String(32))
    role: Mapped[str] = mapped_column(String(16))  # champion | challenger | retired
    status: Mapped[str] = mapped_column(String(16), default="ONAYLI")
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    card: Mapped[dict[str, Any]] = mapped_column(default=dict)
    artifact_path: Mapped[str] = mapped_column(Text, default="")
    approvals: Mapped[list[Any]] = mapped_column(default=list)


class RuleSet(TimestampMixin, Base):
    __tablename__ = "rule_sets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="TASLAK")  # TASLAK|YURURLUKTE|ARSIV
    backtest: Mapped[dict[str, Any]] = mapped_column(default=dict)
    approvals: Mapped[list[Any]] = mapped_column(default=list)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LLMCall(TimestampMixin, Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task: Mapped[str] = mapped_column(String(48))
    mode: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    error_kind: Mapped[str | None] = mapped_column(String(24))
    application_id: Mapped[str | None] = mapped_column(String(20), index=True)


class AuditLog(Base):
    """Hash-chained, actor-attributed audit trail."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)


class OutboxMessage(TimestampMixin, Base):
    __tablename__ = "outbox"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_outbox_idempotency"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    event: Mapped[str] = mapped_column(String(64), index=True)
    aggregate_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(16), default="webhook")
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LoanPerformance(Base):
    """Monthly behaviour of disbursed loans (synthetic) for early warning."""

    __tablename__ = "loan_performance"
    __table_args__ = (Index("ix_perf_app_month", "application_id", "month"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"))
    month: Mapped[int] = mapped_column(Integer)
    dpd: Mapped[int] = mapped_column(Integer, default=0)
    balance: Mapped[float] = mapped_column(Float)
    salary_credited: Mapped[bool] = mapped_column(Boolean, default=True)
    new_bureau_delinquency: Mapped[bool] = mapped_column(Boolean, default=False)
    utilisation: Mapped[float] = mapped_column(Float, default=0.0)
