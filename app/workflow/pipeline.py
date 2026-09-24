"""Staged application pipeline.

``GONDERILDI → (document gate) → BELGE_INCELEMEDE → VERI_TOPLANIYOR →
KARAR_MOTORU → outcome``. Each stage commits, so the API (and a Celery worker
in another process) always sees the current state from the database.

* Missing documents stop the flow in ``BELGE_BEKLENIYOR`` with a letter.
* Provider outages (open circuit, 5xx after retries) leave the application in
  ``VERI_TOPLANIYOR`` with ``retry_count`` incremented; it is retried later.
* LLM failures never fail an application (per-call fallback).
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.llm_service import LLMService
from app.agents.narrator import Narrator
from app.cashflow.analysis import analyse
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import DECISIONS, STAGE_SECONDS
from app.core.rules import load_policy_file, load_pricing, load_workflow
from app.db import outbox
from app.db.models import (
    Application,
    BureauReport,
    CashflowFeatures,
    Decision,
    Document,
    ExtractedField,
    Offer,
    RuleSet,
    utcnow,
)
from app.db.session import session_scope
from app.decisioning.engine import REFERENCE_RATE, DecisionResult, decide
from app.decisioning.features import build_snapshot
from app.decisioning.models import get_models
from app.documents.extraction import parse_amount
from app.integrations.circuit_breaker import CircuitOpenError
from app.integrations.clients import (
    ConsentRequiredError,
    GIBClient,
    KKBClient,
    OpenBankingClient,
    SGKClient,
)
from app.kyc.anomaly import anomaly_score
from app.kyc.network import ring_for
from app.kyc.sanctions import screen_name
from app.kyc.tckn import is_valid_tckn
from app.reports.credit_report import generate_report
from app.workflow.narratives import build_context, missing_docs_context
from app.workflow.service import ApplicationService, age_on
from app.workflow.states import State

logger = get_logger("workflow.pipeline")


class DataCollectionError(RuntimeError):
    """A provider was unavailable; the application stays in VERI_TOPLANIYOR."""


def active_policy(session: Session):
    """The rule set in force: DB-activated version, else the shipped file."""
    from app.core.rules import load_policy_file, parse_policy

    row = session.execute(
        select(RuleSet).where(RuleSet.status == "YURURLUKTE")
    ).scalar_one_or_none()
    if row is not None:
        return parse_policy(row.content)
    return load_policy_file()


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class Pipeline:
    def __init__(self, settings: Settings | None = None, llm: LLMService | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm

    def narrator(self) -> Narrator:
        return Narrator(self.llm or LLMService(self.settings))

    # ------------------------------------------------------------ entry
    async def run(self, application_id: str) -> str:
        """Advance the application as far as possible; return the final state."""
        for _ in range(8):
            with session_scope(self.settings) as session:
                app = session.get(Application, application_id)
                if app is None:
                    raise KeyError(application_id)
                state = State(app.state)
                service = ApplicationService(session, None, self.settings)
                started = time.perf_counter()
                if state in (State.GONDERILDI, State.BELGE_BEKLENIYOR):
                    advanced = await self._document_gate(service, app)
                    STAGE_SECONDS.labels(stage="document_gate").observe(
                        time.perf_counter() - started
                    )
                    if not advanced:
                        return app.state
                elif state == State.BELGE_INCELEMEDE:
                    self._document_review(service, app)
                    STAGE_SECONDS.labels(stage="document_review").observe(
                        time.perf_counter() - started
                    )
                elif state == State.VERI_TOPLANIYOR:
                    try:
                        await self._collect(service, app)
                    except DataCollectionError as exc:
                        app.retry_count += 1
                        app.last_error = str(exc)
                        service.audit(
                            "DATA_COLLECTION_RETRY",
                            app.id,
                            {"error": str(exc), "retry": app.retry_count},
                        )
                        return app.state
                    STAGE_SECONDS.labels(stage="data_collection").observe(
                        time.perf_counter() - started
                    )
                elif state == State.KARAR_MOTORU:
                    await self._decide(service, app)
                    STAGE_SECONDS.labels(stage="decision").observe(time.perf_counter() - started)
                else:
                    return app.state
        return state.value

    # ------------------------------------------------------------ stages
    async def _document_gate(self, service: ApplicationService, app: Application) -> bool:
        missing = service.missing_documents(app)
        if missing:
            if app.state != State.BELGE_BEKLENIYOR.value:
                pii = service.pii(app)
                letter = await self.narrator().missing_documents_letter(
                    missing_docs_context(app, pii["name"], missing, pii)
                )
                service.transition(
                    app, State.BELGE_BEKLENIYOR, "eksik belge: " + ", ".join(c for c, _ in missing)
                )
                app.letters = {
                    **(app.letters or {}),
                    "eksik_belge": letter.text,
                    "eksik_belge_mode": letter.mode,
                }
                outbox.enqueue(
                    service.session,
                    event="EKSIK_BELGE_YAZISI",
                    aggregate_id=app.id,
                    channel="email",
                    payload={
                        "to": pii.get("email") or "",
                        "subject": f"{app.id} eksik belge",
                        "body": letter.text,
                    },
                    idempotency_key=f"{app.id}:missing-docs:{len(app.events)}",
                )
            return False
        service.transition(app, State.BELGE_INCELEMEDE, "belgeler tamamlandı")
        return True

    def _document_review(self, service: ApplicationService, app: Application) -> None:
        documents = service.documents(app)
        for document in documents:
            if document.status == "YUKLENDI":
                service.process_document(document)
        service.transition(app, State.VERI_TOPLANIYOR, "belge incelemesi tamamlandı")

    async def _collect(self, service: ApplicationService, app: Application) -> None:
        s = self.settings
        pii = service.pii(app)
        tckn = pii["identity_no"]
        consents = service.active_consents(app)
        income = app.declared_income
        kkb, sgk_client = KKBClient(s), SGKClient(s)
        bureau: dict[str, Any] | None = None
        sgk: dict[str, Any] | None = None
        gib: dict[str, Any] | None = None
        transactions: dict[str, Any] | None = None
        try:
            bureau, sgk = await asyncio.gather(
                kkb.get_report(tckn, income_hint=income, consents=consents),
                sgk_client.get_record(
                    tckn, income_hint=income, employment_type=app.employment_type, consents=consents
                ),
            )
            if app.employment_type == "SERBEST":
                gib = await GIBClient(s).get_tax_record(tckn, income_hint=income, consents=consents)
            try:
                transactions = await OpenBankingClient(s).get_transactions(
                    tckn, income_hint=income, consents=consents
                )
            except ConsentRequiredError:
                transactions = None  # open banking is optional: decision without cash flow
        except ConsentRequiredError as exc:
            raise DataCollectionError(f"rıza eksik: {exc}") from exc
        except CircuitOpenError as exc:
            raise DataCollectionError(f"servis devre dışı: {exc}") from exc
        except Exception as exc:  # provider outage after retries
            raise DataCollectionError(f"entegrasyon hatası: {type(exc).__name__}") from exc
        session = service.session
        for provider, payload in (("KKB", bureau), ("SGK", sgk), ("GIB", gib)):
            if payload is not None:
                session.add(BureauReport(application_id=app.id, provider=provider, payload=payload))
        cash_features: dict[str, float] = {}
        if transactions is not None:
            result = analyse(
                transactions["transactions"], transactions["account"]["opening_balance"]
            )
            cash_features = result.features
            session.add(
                CashflowFeatures(
                    application_id=app.id,
                    features=result.features,
                    monthly=[m.model_dump() for m in result.monthly],
                    categories=result.category_totals,
                )
            )
        app.kyc = self._kyc(service, app, pii)
        app.kyc = {**app.kyc, "reconciliation": self._reconcile(service, app, cash_features, sgk)}
        app.retry_count = 0
        app.last_error = None
        service.transition(app, State.KARAR_MOTORU, "veri toplama tamamlandı")

    def _kyc(
        self, service: ApplicationService, app: Application, pii: dict[str, Any]
    ) -> dict[str, Any]:
        workflow = load_workflow()
        screening = screen_name(pii["name"])
        service.session.flush()
        ring = ring_for(app.id, service.identity_records(), workflow.fraud_ring_min_size)
        return {
            "tckn_valid": is_valid_tckn(pii["identity_no"]),
            "sanctions_match": screening.matched,
            "sanctions": screening.model_dump(),
            "velocity_30d": service.velocity(app),
            "fraud_ring_size": ring.ring_size if ring.flagged else 1,
            "ring": ring.model_dump(),
            "anomaly_score": round(
                anomaly_score(
                    income=app.declared_income,
                    amount=app.requested_amount,
                    term_months=app.requested_term_months,
                    age=age_on(app.applicant.birth_date),
                ),
                4,
            ),
        }

    def _reconcile(
        self,
        service: ApplicationService,
        app: Application,
        cash: dict[str, float],
        sgk: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tolerance = load_workflow().income_tolerance
        declared = app.declared_income
        fields = (
            service.session.execute(
                select(ExtractedField).where(
                    ExtractedField.application_id == app.id, ExtractedField.name == "net_ucret"
                )
            )
            .scalars()
            .all()
        )
        payslip = None
        if fields:
            try:
                payslip = parse_amount(fields[-1].value)
            except ValueError:
                payslip = None
        bank = cash.get("avg_monthly_income") if cash else None
        checks = {}
        for label, value in (("bordro", payslip), ("hesap", bank)):
            if value:
                checks[label] = {
                    "value": round(value, 2),
                    "deviation": round(abs(value - declared) / declared, 4),
                }
        mismatch = any(c["deviation"] > tolerance for c in checks.values())
        if payslip and bank and abs(payslip - bank) / max(payslip, 1) > tolerance:
            mismatch = True
        confidences = [
            f.confidence
            for f in service.session.execute(
                select(ExtractedField).where(ExtractedField.application_id == app.id)
            ).scalars()
        ]
        documents = service.documents(app)
        if any(d.status == "OCR_GEREKLI" for d in documents):
            confidences.append(0.0)
        verified = bank if bank and not mismatch else None
        return {
            "declared": declared,
            "checks": checks,
            "tolerance": tolerance,
            "income_mismatch": mismatch,
            "verified_income": round(verified, 2) if verified else None,
            "min_field_confidence": round(min(confidences), 3) if confidences else 1.0,
            "max_fraud_score": max((d.fraud_score for d in documents), default=0.0),
            "fraud_documents": [d.code for d in documents if d.fraud_score >= 0.6],
        }

    def build_snapshot(self, service: ApplicationService, app: Application) -> dict[str, Any]:
        session = service.session
        reports = {
            r.provider: r.payload
            for r in session.execute(
                select(BureauReport)
                .where(BureauReport.application_id == app.id)
                .order_by(BureauReport.id)
            ).scalars()
        }
        cash = (
            session.execute(
                select(CashflowFeatures)
                .where(CashflowFeatures.application_id == app.id)
                .order_by(CashflowFeatures.id.desc())
            )
            .scalars()
            .first()
        )
        recon = (app.kyc or {}).get("reconciliation", {})
        return build_snapshot(
            product=app.product,
            requested_amount=app.requested_amount,
            term_months=app.requested_term_months,
            declared_income=app.declared_income,
            verified_income=None,
            age=age_on(app.applicant.birth_date),
            reference_rate=REFERENCE_RATE,
            bureau=reports.get("KKB"),
            sgk=reports.get("SGK"),
            cashflow=cash.features if cash else None,
            kyc=app.kyc or {},
            documents={
                "max_fraud_score": recon.get("max_fraud_score", 0.0),
                "income_mismatch": recon.get("income_mismatch", False),
                "min_field_confidence": recon.get("min_field_confidence", 1.0),
            },
        )

    async def _decide(self, service: ApplicationService, app: Application) -> None:
        session = service.session
        policy = active_policy(session)
        snapshot = self.build_snapshot(service, app)
        result = decide(snapshot, policy=policy, models=get_models(), pricing_cfg=load_pricing())
        decision = persist_decision(session, app, snapshot, result)
        DECISIONS.labels(outcome=result.outcome).inc()
        pii = service.pii(app)
        flags = [s["label"] for d in service.documents(app) for s in d.fraud_signals]
        ctx = build_context(
            app, decision, applicant_name=pii["name"], pii=pii, fraud_flags=flags[:5]
        )
        bundle = await self.narrator().run_chain(ctx)
        decision.narratives = {
            "committee_summary": bundle.committee_summary,
            "applicant_letter": bundle.applicant_letter,
            "analysis": bundle.analysis.model_dump(),
            "mode": bundle.overall_mode,
            "modes": bundle.modes,
            "errors": bundle.errors,
        }
        # Slow work (narratives) happens before the first audit append: the audit
        # chain lock is held from the first append until commit.
        service.audit(
            "DECISION_MADE",
            app.id,
            {
                "decision_id": decision.id,
                "outcome": result.outcome,
                "pd": result.pd,
                "rule_set": decision.rule_set_version,
                "model": decision.model_version,
                "feature_hash": decision.feature_hash,
            },
        )
        app.decided_at = utcnow()
        target = State(result.outcome)
        service.transition(
            app,
            target,
            "karar motoru: " + ", ".join(result.reason_code_list)
            if result.reason_code_list
            else "karar motoru",
        )
        offer: Offer | None = None
        if target == State.OTOMATIK_ONAY and result.pricing is not None:
            offer = create_offer(session, app, decision)
            service.transition(app, State.TEKLIF_SUNULDU, "teklif oluşturuldu")
        elif target == State.UZMAN_INCELEMESI:
            app.priority = queue_priority(app, decision)
        app.letters = {**(app.letters or {}), "karar": bundle.applicant_letter}
        app.reports = generate_report(app, decision, offer, pii, self.settings)
        outbox.enqueue(
            session,
            event="KARAR_BILDIRIMI",
            aggregate_id=app.id,
            channel="email",
            payload={
                "to": pii.get("email") or "",
                "subject": f"{app.id} kredi başvurunuz hakkında",
                "body": bundle.applicant_letter,
            },
            idempotency_key=f"{app.id}:decision:{decision.id}",
        )


def persist_decision(
    session: Session,
    app: Application,
    snapshot: dict[str, Any],
    result: DecisionResult,
    *,
    kind: str = "engine",
    decided_by: str = "engine",
) -> Decision:
    decision = Decision(
        application_id=app.id,
        kind=kind,
        outcome=result.outcome,
        conditional=result.conditional,
        pd=result.pd,
        score_points=result.score.get("points"),
        risk_band=result.risk_band,
        reason_codes=[r.model_dump() for r in result.reason_codes],
        rule_set_version=result.versions["rule_set"],
        model_version=result.versions["model"],
        feature_snapshot=snapshot,
        feature_hash=result.feature_hash,
        rule_results=[r.model_dump() for r in result.rule_results],
        limits=result.limits,
        pricing=result.pricing.model_dump() if result.pricing else {},
        counterfactuals=[c.model_dump() for c in result.counterfactuals],
        explanation={
            **result.explanation,
            "scorecard": result.score,
            "pd_requested": result.pd_requested,
        },
        challenger=result.challenger,
        latency_ms=result.latency_ms,
        decided_by=decided_by,
    )
    session.add(decision)
    session.flush()
    return decision


def queue_priority(app: Application, decision: Decision) -> float:
    weights = load_workflow().queue_priority
    amount_score = min(app.requested_amount / 1_000_000, 1.0)
    risk_score = min((decision.pd or 0.0) / 0.2, 1.0)
    return round(
        weights.get("amount_weight", 0.4) * amount_score
        + weights.get("risk_weight", 0.4) * risk_score
        + weights.get("sla_weight", 0.2),
        4,
    )


def create_offer(
    session: Session, app: Application, decision: Decision, *, pricing: dict[str, Any] | None = None
) -> Offer:
    data = pricing or decision.pricing
    cfg = load_pricing()
    offer = Offer(
        application_id=app.id,
        decision_id=decision.id,
        amount=data["amount"],
        term_months=data["term_months"],
        annual_rate=data["annual_rate"],
        instalment=data["instalment"],
        apr=data["apr"],
        total_payment=data["total_payment"],
        fees=data.get("upfront_fee", 0.0),
        details={
            "schedule": data.get("schedule", []),
            "total_taxes": data.get("total_taxes"),
            "gross_monthly_rate": data.get("gross_monthly_rate"),
        },
        valid_until=utcnow() + timedelta(days=cfg.offer_validity_days),
    )
    session.add(offer)
    session.flush()
    return offer


def latest_decision(session: Session, application_id: str) -> Decision | None:
    return (
        session.execute(
            select(Decision)
            .where(Decision.application_id == application_id)
            .order_by(Decision.created_at.desc(), Decision.id.desc())
        )
        .scalars()
        .first()
    )


def latest_offer(session: Session, application_id: str) -> Offer | None:
    return (
        session.execute(
            select(Offer)
            .where(Offer.application_id == application_id)
            .order_by(Offer.created_at.desc())
        )
        .scalars()
        .first()
    )


def replay_decision(session: Session, decision: Decision) -> dict[str, Any]:
    """Re-run the engine on the stored snapshot and compare."""
    from app.core.rules import parse_policy

    policy = load_policy_file()
    if decision.rule_set_version != policy.version:
        row = session.get(RuleSet, decision.rule_set_version)
        if row is not None:
            policy = parse_policy(row.content)
    models = get_models()
    result = decide(
        decision.feature_snapshot, policy=policy, models=models, pricing_cfg=load_pricing()
    )
    same = (
        result.outcome == decision.outcome
        and abs(result.pd - (decision.pd or 0.0)) < 1e-9
        and [r.code for r in result.reason_codes] == [r["code"] for r in decision.reason_codes]
        and result.feature_hash == decision.feature_hash
    )
    return {
        "decision_id": decision.id,
        "identical": same,
        "original": {
            "outcome": decision.outcome,
            "pd": decision.pd,
            "reason_codes": [r["code"] for r in decision.reason_codes],
        },
        "replayed": {
            "outcome": result.outcome,
            "pd": result.pd,
            "reason_codes": result.reason_code_list,
        },
        "versions": {
            "rule_set": decision.rule_set_version,
            "model": decision.model_version,
            "model_available": models.pd_model.version,
        },
        "feature_hash": decision.feature_hash,
    }


def run_pipeline(application_id: str, settings: Settings | None = None) -> str:
    """Synchronous entry point used by the inline dispatcher and Celery."""
    return _run(Pipeline(settings).run(application_id))


def documents_of(session: Session, application_id: str) -> list[Document]:
    return list(
        session.execute(select(Document).where(Document.application_id == application_id)).scalars()
    )
