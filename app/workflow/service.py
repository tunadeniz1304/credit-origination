"""Application service: every state change goes through here.

Transitions are validated by the state machine and each one writes an
``application_events`` row, a hash-chained audit entry, a Prometheus counter
and — for customer-visible milestones — a transactional outbox message, all in
the caller's transaction.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core import crypto
from app.core.config import Settings, get_settings, load_pipeline_rules
from app.core.logging import get_logger
from app.core.metrics import APPLICATIONS_SUBMITTED, STATE_TRANSITIONS
from app.core.rules import load_workflow
from app.core.security import Principal
from app.db import outbox
from app.db.audit import append_audit
from app.db.models import (
    Applicant,
    Application,
    ApplicationEvent,
    Consent,
    Document,
    ExtractedField,
    utcnow,
)
from app.documents.extraction import extract_fields, extract_text
from app.documents.fraud import analyse_document
from app.documents.storage import store_upload
from app.workflow.states import State, assert_transition

logger = get_logger("workflow.service")

SYSTEM = "system"
NOTIFY_STATES = {
    State.BELGE_BEKLENIYOR: "BELGE_BEKLENIYOR",
    State.OTOMATIK_ONAY: "KARAR_ONAY",
    State.OTOMATIK_RET: "KARAR_RET",
    State.UZMAN_INCELEMESI: "UZMAN_INCELEMESINDE",
    State.TEKLIF_SUNULDU: "TEKLIF_SUNULDU",
    State.REDDEDILDI: "KARAR_RET",
    State.ITIRAZ_INCELEMESI: "ITIRAZ_ALINDI",
    State.SOZLESME_HAZIR: "SOZLESME_HAZIR",
    State.KULLANDIRILDI: "KULLANDIRILDI",
}
CONSENT_TYPES = ("KVKK_AYDINLATMA", "ACIK_RIZA", "KKB_SORGU", "EDEVLET_SORGU", "ACIK_BANKACILIK")


def new_application_id() -> str:
    return f"APP-{uuid.uuid4().hex[:12].upper()}"


def actor_name(principal: Principal | None) -> str:
    return principal.username if principal else SYSTEM


def required_documents(employment_type: str) -> list[tuple[str, str]]:
    policy = load_pipeline_rules().document_policy
    docs = [(d.code, d.description) for d in policy.required_documents]
    if employment_type == "SERBEST":
        docs += [(d.code, d.description) for d in policy.self_employed_documents]
    return docs


def age_on(birth: date | None, today: date | None = None) -> int:
    if birth is None:
        return 35
    today = today or datetime.now(UTC).date()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


class ApplicationService:
    """Unit-of-work scoped service (one session, one actor)."""

    def __init__(
        self, session: Session, actor: Principal | None = None, settings: Settings | None = None
    ) -> None:
        self.session = session
        self.actor = actor
        self.settings = settings or get_settings()

    # ------------------------------------------------------------ transitions
    def transition(self, app: Application, target: State, reason: str = "") -> None:
        current = State(app.state)
        assert_transition(current, target)
        app.state = target.value
        app.updated_at = utcnow()
        self.session.add(
            ApplicationEvent(
                application_id=app.id,
                from_state=current.value,
                to_state=target.value,
                actor=actor_name(self.actor),
                reason=reason,
            )
        )
        append_audit(
            self.session,
            actor=actor_name(self.actor),
            action=f"STATE_{target.value}",
            entity_type="application",
            entity_id=app.id,
            payload={"from": current.value, "to": target.value, "reason": reason},
        )
        STATE_TRANSITIONS.labels(to_state=target.value).inc()
        if target in NOTIFY_STATES:
            outbox.enqueue(
                self.session,
                event=NOTIFY_STATES[target],
                aggregate_id=app.id,
                payload={"application_id": app.id, "state": target.value, "reason": reason},
                idempotency_key=f"{app.id}:{target.value}:{len(app.events) + 1}",
            )
        sla = load_workflow().sla_hours.get(target.value)
        if sla:
            app.sla_due_at = utcnow() + timedelta(hours=sla)
        logger.info("application %s: %s -> %s", app.id, current.value, target.value)

    def audit(
        self,
        action: str,
        entity_id: str,
        payload: dict[str, Any] | None = None,
        entity_type: str = "application",
    ) -> None:
        append_audit(
            self.session,
            actor=actor_name(self.actor),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload or {},
        )

    # ------------------------------------------------------------ creation
    def create(self, data: dict[str, Any]) -> Application:
        """Create applicant + application, record consents and submit."""
        s = self.settings
        applicant = Applicant(
            owner_user_id=self.actor.user_id
            if self.actor and self.actor.role == "basvuran"
            else None,
            tckn_enc=crypto.encrypt(data["identity_no"], s),
            tckn_bidx=crypto.blind_index(data["identity_no"], s),
            name_enc=crypto.encrypt(data["name"], s),
            phone_enc=crypto.encrypt(data.get("phone"), s),
            phone_bidx=crypto.blind_index(data.get("phone"), s),
            email_enc=crypto.encrypt(data.get("email"), s),
            address_enc=crypto.encrypt(data.get("address"), s),
            address_bidx=crypto.blind_index(data.get("address"), s),
            iban_enc=crypto.encrypt(data.get("iban"), s),
            iban_bidx=crypto.blind_index(data.get("iban"), s),
            birth_date=data.get("birth_date"),
            gender=data.get("gender"),
            province=data.get("province"),
        )
        self.session.add(applicant)
        self.session.flush()
        app = Application(
            id=new_application_id(),
            applicant_id=applicant.id,
            owner_user_id=applicant.owner_user_id,
            state=State.TASLAK.value,
            product=data.get("product", "IHTIYAC"),
            requested_amount=float(data["requested_amount"]),
            requested_term_months=int(data["requested_term_months"]),
            currency=data.get("currency", "TRY"),
            declared_income=float(data["monthly_income"]),
            employment_type=data.get("employment_type", "MAASLI"),
            employer_name=data.get("employer_name"),
            device_id=data.get("device_id"),
            persona=data.get("persona"),
        )
        self.session.add(app)
        self.session.flush()
        self.session.add(
            ApplicationEvent(
                application_id=app.id,
                from_state=None,
                to_state=State.TASLAK.value,
                actor=actor_name(self.actor),
                reason="taslak oluşturuldu",
            )
        )
        self.audit(
            "APPLICATION_CREATED",
            app.id,
            {
                "product": app.product,
                "amount": app.requested_amount,
                "term": app.requested_term_months,
            },
        )
        for consent_type, granted in (data.get("consents") or {}).items():
            if granted:
                self.grant_consent(app, consent_type.upper())
        self.transition(app, State.GONDERILDI, "başvuru gönderildi")
        app.submitted_at = utcnow()
        APPLICATIONS_SUBMITTED.inc()
        return app

    # ------------------------------------------------------------ consents
    def grant_consent(
        self,
        app: Application,
        consent_type: str,
        *,
        scope: list[str] | None = None,
        days: int = 365,
    ) -> Consent:
        if consent_type not in CONSENT_TYPES:
            raise ValueError(f"unknown consent type {consent_type}")
        consent = Consent(
            application_id=app.id,
            type=consent_type,
            scope=scope or [],
            expires_at=utcnow() + timedelta(days=days),
        )
        self.session.add(consent)
        self.audit(
            "CONSENT_GRANTED", app.id, {"type": consent_type, "scope": scope or [], "days": days}
        )
        return consent

    def active_consents(self, app: Application) -> list[str]:
        now = utcnow()
        rows = self.session.execute(
            select(Consent).where(Consent.application_id == app.id)
        ).scalars()
        active = []
        for consent in rows:
            expires = consent.expires_at
            if expires is not None and expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            if consent.revoked_at is None and (expires is None or expires > now):
                active.append(consent.type)
        return active

    # ------------------------------------------------------------ documents
    def add_document(self, app: Application, code: str, filename: str, data: bytes) -> Document:
        code = code.upper()
        known = {c for c, _ in required_documents(app.employment_type)} | {"TAX_PLATE", "OTHER"}
        if code not in known:
            raise ValueError(f"unknown document code {code}")
        stored = store_upload(app.id, code, filename, data, self.settings)
        document = Document(
            application_id=app.id,
            code=code,
            filename=stored.filename,
            stored_path=stored.path,
            sha256=stored.sha256,
            mime=stored.mime,
            size=stored.size,
            uploaded_by=actor_name(self.actor),
        )
        self.session.add(document)
        self.session.flush()
        self.process_document(document)
        self.audit(
            "DOCUMENT_UPLOADED",
            app.id,
            {
                "document_id": document.id,
                "code": code,
                "sha256": stored.sha256,
                "fraud_score": document.fraud_score,
            },
        )
        return document

    def analyse_file(self, document: Document) -> dict[str, Any]:
        """Pure analysis of a stored document (no database writes)."""
        path = Path(document.stored_path)
        text = extract_text(path, document.mime)
        if text.source == "none":
            return {"source": "none", "fields": [], "report": None}
        fields = extract_fields(document.code, text, path)
        report = analyse_document(path, document.code, document.mime, fields, text)
        return {"source": text.source, "fields": fields, "report": report}

    def apply_analysis(self, document: Document, analysis: dict[str, Any]) -> None:
        document.text_source = analysis["source"]
        report = analysis["report"]
        if report is None:
            document.status = "OCR_GEREKLI"
            document.fraud_score = 0.0
            document.fraud_signals = []
            return
        document.fraud_score = report.score
        document.fraud_signals = [s.model_dump() for s in report.signals]
        document.status = "SUPHELI" if report.score >= 0.6 else "ISLENDI"
        for f in analysis["fields"]:
            self.session.add(
                ExtractedField(
                    document_id=document.id,
                    application_id=document.application_id,
                    name=f.name,
                    value=f.value,
                    confidence=f.confidence,
                    page=f.page,
                    bbox=f.bbox,
                    source=f.source,
                )
            )

    def process_document(self, document: Document) -> None:
        self.apply_analysis(document, self.analyse_file(document))

    def documents(self, app: Application) -> list[Document]:
        return list(
            self.session.execute(
                select(Document)
                .where(Document.application_id == app.id)
                .order_by(Document.created_at)
            ).scalars()
        )

    def missing_documents(self, app: Application) -> list[tuple[str, str]]:
        present = {d.code for d in self.documents(app) if d.status != "REDDEDILDI"}
        return [
            (code, desc)
            for code, desc in required_documents(app.employment_type)
            if code not in present
        ]

    # ------------------------------------------------------------ KYC helpers
    def velocity(self, app: Application) -> int:
        window = utcnow() - timedelta(days=load_workflow().velocity_window_days)
        applicant = app.applicant
        conditions = [Applicant.tckn_bidx == applicant.tckn_bidx]
        if applicant.phone_bidx:
            conditions.append(Applicant.phone_bidx == applicant.phone_bidx)
        if applicant.iban_bidx:
            conditions.append(Applicant.iban_bidx == applicant.iban_bidx)
        query = (
            select(func.count(Application.id))
            .join(Applicant, Applicant.id == Application.applicant_id)
            .where(Application.id != app.id, Application.created_at >= window, or_(*conditions))
        )
        count = self.session.execute(query).scalar_one()
        if app.device_id:
            count += self.session.execute(
                select(func.count(Application.id)).where(
                    Application.id != app.id,
                    Application.device_id == app.device_id,
                    Application.created_at >= window,
                    Application.applicant_id != app.applicant_id,
                )
            ).scalar_one()
        return int(count)

    def identity_records(self, limit: int = 1000) -> list[Any]:
        from app.kyc.network import IdentityRecord

        rows = self.session.execute(
            select(
                Application.id,
                Application.device_id,
                Applicant.tckn_bidx,
                Applicant.phone_bidx,
                Applicant.iban_bidx,
                Applicant.address_bidx,
            )
            .join(Applicant, Applicant.id == Application.applicant_id)
            .order_by(Application.created_at.desc())
            .limit(limit)
        ).all()
        return [
            IdentityRecord(
                application_id=r[0],
                applicant_key=r[2] or r[0],
                identifiers={"device": r[1], "phone": r[3], "iban": r[4], "address": r[5]},
            )
            for r in rows
        ]

    # ------------------------------------------------------------ PII views
    def pii(self, app: Application) -> dict[str, Any]:
        a = app.applicant
        s = self.settings
        return {
            "name": crypto.decrypt(a.name_enc, s) or "",
            "identity_no": crypto.decrypt(a.tckn_enc, s) or "",
            "phone": crypto.decrypt(a.phone_enc, s),
            "email": crypto.decrypt(a.email_enc, s),
            "address": crypto.decrypt(a.address_enc, s),
            "iban": crypto.decrypt(a.iban_enc, s),
        }
