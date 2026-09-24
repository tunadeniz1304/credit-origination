"""Underwriter workbench: queue, overrides, maker-checker, objections."""

from __future__ import annotations

from datetime import UTC
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.agents.llm_service import LLMService
from app.agents.narrator import Narrator
from app.core.labels import label
from app.core.security import Principal
from app.db.models import Application, Decision, Objection, Review, utcnow
from app.pricing.engine import quote
from app.workbench.authority import checker_allowed, evaluate_authority
from app.workflow.narratives import build_context
from app.workflow.pipeline import create_offer, latest_decision
from app.workflow.service import ApplicationService
from app.workflow.states import State

REVIEWABLE = (State.UZMAN_INCELEMESI.value, State.ITIRAZ_INCELEMESI.value)
URGENT_SLA_HOURS = 4.0  # queue boost when the SLA runs out within this window


class WorkbenchError(ValueError):
    """Business-rule violation in the workbench (maps to 403/409)."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def sla_remaining_hours(app: Application) -> float | None:
    if app.sla_due_at is None:
        return None
    due = app.sla_due_at if app.sla_due_at.tzinfo else app.sla_due_at.replace(tzinfo=UTC)
    return round((due - utcnow()).total_seconds() / 3600, 1)


def _queue_key(app: Application) -> tuple[float, float]:
    remaining = sla_remaining_hours(app)
    urgency = 1.0 if remaining is not None and remaining < URGENT_SLA_HOURS else 0.0
    return (-((app.priority or 0.0) + urgency), remaining if remaining is not None else 1e9)


def queue(
    session: Session,
    *,
    state: str | None = None,
    assigned_to: str | None = None,
    search: str | None = None,
    product: str | None = None,
    sla_breached: bool | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    """Filtered, prioritised review queue with server-side pagination.

    Filtering and ordering use only application columns; the latest decision
    and pending review are loaded for the returned page only.
    """
    query = select(Application).where(Application.state.in_(REVIEWABLE))
    if state:
        query = query.where(Application.state == state)
    if assigned_to:
        query = query.where(Application.assigned_to == assigned_to)
    if product:
        query = query.where(Application.product == product)
    if search:
        term = f"%{search.strip()}%"
        query = query.where(or_(Application.id.ilike(term), Application.persona.ilike(term)))
    apps = list(session.execute(query).scalars())
    if sla_breached is not None:
        apps = [
            a
            for a in apps
            if ((sla_remaining_hours(a) or 0.0) < 0 and a.sla_due_at is not None) == sla_breached
        ]
    apps.sort(key=_queue_key)
    total = len(apps)
    breached = sum(
        1 for a in apps if a.sla_due_at is not None and (sla_remaining_hours(a) or 0) < 0
    )
    page = apps[offset : offset + limit]
    ids = [a.id for a in page]
    pending_by_app = {
        r.application_id: r.id
        for r in session.execute(
            select(Review).where(Review.application_id.in_(ids), Review.status == "ONAY_BEKLIYOR")
        ).scalars()
    }
    items = []
    for app in page:
        decision = latest_decision(session, app.id)
        remaining = sla_remaining_hours(app)
        urgency = 1.0 if remaining is not None and remaining < URGENT_SLA_HOURS else 0.0
        items.append(
            {
                "application_id": app.id,
                "state": app.state,
                "requested_amount": app.requested_amount,
                "product": app.product,
                "product_label": label("product", app.product),
                "pd": decision.pd if decision else None,
                "risk_band": decision.risk_band if decision else None,
                "reason_codes": [r["code"] for r in decision.reason_codes] if decision else [],
                "sla_due_at": app.sla_due_at.isoformat() if app.sla_due_at else None,
                "sla_remaining_hours": remaining,
                "sla_breached": remaining is not None and remaining < 0,
                "assigned_to": app.assigned_to,
                "priority": round((app.priority or 0.0) + urgency, 4),
                "pending_review_id": pending_by_app.get(app.id),
                "persona": app.persona,
            }
        )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "sla_breached": breached,
    }


class Workbench:
    def __init__(self, session: Session, user: Principal) -> None:
        self.session = session
        self.user = user
        self.service = ApplicationService(session, user)

    def assign(self, app: Application, assignee: str | None) -> None:
        if app.state not in REVIEWABLE:
            raise WorkbenchError("başvuru inceleme kuyruğunda değil")
        app.assigned_to = assignee or self.user.username
        self.service.audit("REVIEW_ASSIGNED", app.id, {"assigned_to": app.assigned_to})

    def submit_decision(
        self,
        app: Application,
        *,
        action: str,
        justification: str,
        amount: float | None = None,
        term_months: int | None = None,
        annual_rate: float | None = None,
    ) -> Review:
        if app.state not in REVIEWABLE:
            raise WorkbenchError("başvuru inceleme aşamasında değil")
        if (
            self.session.execute(
                select(Review).where(
                    Review.application_id == app.id, Review.status == "ONAY_BEKLIYOR"
                )
            )
            .scalars()
            .first()
        ):
            raise WorkbenchError("bu başvuru için onay bekleyen bir karar var")
        engine = latest_decision(self.session, app.id)
        pd = engine.pd if engine and engine.pd is not None else 0.1
        engine_offer = (engine.limits or {}).get("offer_amount") if engine else None
        final_amount = amount or engine_offer or app.requested_amount
        final_term = (
            term_months or (engine.limits or {}).get("term_months") if engine else term_months
        )
        final_term = final_term or app.requested_term_months
        engine_rate = (engine.pricing or {}).get("annual_rate") if engine else None
        is_override = False
        if engine is not None:
            if engine.outcome == "OTOMATIK_RET" and action == "ONAY":
                is_override = True
            if engine.outcome == "OTOMATIK_ONAY" and action == "RET":
                is_override = True
            if action == "ONAY" and engine_offer and final_amount > engine_offer:
                is_override = True
            if (
                action == "ONAY"
                and annual_rate is not None
                and engine_rate
                and abs(annual_rate - engine_rate) > 1e-6
            ):
                is_override = True
        check = evaluate_authority(
            self.user,
            amount=final_amount if action == "ONAY" else app.requested_amount,
            pd=pd,
            is_override=is_override,
        )
        if action == "ONAY" and self.user.authority_rank < 1:
            raise WorkbenchError("kredi kararı yetkiniz yok", 403)
        review = Review(
            application_id=app.id,
            maker=self.user.username,
            maker_role=self.user.role,
            action=action,
            amount=final_amount if action == "ONAY" else None,
            term_months=final_term if action == "ONAY" else None,
            annual_rate=annual_rate,
            justification=justification,
            is_override=is_override,
            required_role=check.required_role,
            four_eyes=check.four_eyes,
        )
        self.session.add(review)
        self.session.flush()
        self.service.audit(
            "REVIEW_SUBMITTED",
            app.id,
            {
                "review_id": review.id,
                "action": action,
                "override": is_override,
                "four_eyes": check.four_eyes,
                "required_role": check.required_role,
                "reasons": check.reasons,
            },
        )
        if action == "BELGE_ISTE" or check.maker_may_finalise:
            self._finalise(app, review)
        elif not check.four_eyes:
            raise WorkbenchError(
                f"bu karar için en az '{check.required_role}' yetkisi gerekir", 403
            )
        return review

    def check(self, review: Review, *, approve: bool, note: str) -> Review:
        if review.status != "ONAY_BEKLIYOR":
            raise WorkbenchError("bu karar onay beklemiyor")
        from app.core.security import CREDIT_AUTHORITY_RANK

        allowed, reason = checker_allowed(
            self.user, review.maker, CREDIT_AUTHORITY_RANK[review.required_role]
        )
        if not allowed:
            raise WorkbenchError(reason, 403)
        review.checker = self.user.username
        review.checker_note = note
        review.decided_at = utcnow()
        app = self.session.get(Application, review.application_id)
        assert app is not None
        self.service.audit(
            "REVIEW_CHECKED",
            app.id,
            {"review_id": review.id, "approve": approve, "checker": self.user.username},
        )
        if approve:
            self._finalise(app, review)
        else:
            review.status = "REDDEDILDI"
        return review

    def _finalise(self, app: Application, review: Review) -> None:
        review.status = "TAMAMLANDI"
        review.decided_at = review.decided_at or utcnow()
        engine = latest_decision(self.session, app.id)
        if review.action == "BELGE_ISTE":
            if app.state != State.UZMAN_INCELEMESI.value:
                raise WorkbenchError("ek belge yalnızca uzman incelemesinde istenebilir")
            self.service.transition(
                app, State.BELGE_BEKLENIYOR, "uzman ek belge istedi: " + review.justification[:120]
            )
            app.letters = {**(app.letters or {}), "ek_belge": review.justification}
            return
        pd = engine.pd if engine and engine.pd is not None else 0.1
        kind = "override" if review.is_override else "manual"
        decision = Decision(
            application_id=app.id,
            kind=kind,
            outcome="ONAYLANDI" if review.action == "ONAY" else "REDDEDILDI",
            pd=pd,
            score_points=engine.score_points if engine else None,
            risk_band=engine.risk_band if engine else None,
            reason_codes=engine.reason_codes if engine and review.action == "RET" else [],
            rule_set_version=engine.rule_set_version if engine else "manual",
            model_version=engine.model_version if engine else "manual",
            feature_snapshot=engine.feature_snapshot if engine else {},
            feature_hash=engine.feature_hash if engine else "",
            rule_results=engine.rule_results if engine else [],
            limits=engine.limits if engine else {},
            counterfactuals=engine.counterfactuals if engine and review.action == "RET" else [],
            explanation={
                "review_id": review.id,
                "justification": review.justification,
                "maker": review.maker,
                "checker": review.checker,
            },
            decided_by=review.checker or review.maker,
        )
        if review.action == "ONAY":
            priced = quote(
                pd=pd,
                amount=review.amount or app.requested_amount,
                term_months=review.term_months or app.requested_term_months,
                product=app.product,
                rate_override=review.annual_rate,
            )
            if priced.exceeds_cap:
                raise WorkbenchError("önerilen faiz yasal azami oranı aşıyor")
            decision.pricing = priced.model_dump()
        self.session.add(decision)
        self.session.flush()
        pii = self.service.pii(app)
        ctx = build_context(
            app, decision, applicant_name=pii["name"], pii=pii, outcome=decision.outcome
        )
        from app.core.task_dispatcher import run_coroutine_safe

        letter = run_coroutine_safe(Narrator(LLMService()).applicant_letter(ctx))
        decision.narratives = {"applicant_letter": letter.text, "mode": letter.mode}
        app.letters = {**(app.letters or {}), "karar": letter.text}
        app.decided_at = utcnow()
        if review.action == "ONAY":
            create_offer(self.session, app, decision)
            self.service.transition(
                app, State.TEKLIF_SUNULDU, f"uzman kararı ({kind}): {review.justification[:120]}"
            )
        else:
            self.service.transition(
                app, State.REDDEDILDI, f"uzman kararı ({kind}): {review.justification[:120]}"
            )
        self.service.audit(
            "REVIEW_FINALISED",
            app.id,
            {
                "review_id": review.id,
                "decision_id": decision.id,
                "outcome": decision.outcome,
                "kind": kind,
            },
        )

    def resolve_objection(self, app: Application, *, upheld: bool, note: str) -> Objection:
        if app.state != State.ITIRAZ_INCELEMESI.value:
            raise WorkbenchError("başvuru itiraz incelemesinde değil")
        if self.user.authority_rank < 1:
            raise WorkbenchError("itiraz çözme yetkiniz yok", 403)
        objection = (
            self.session.execute(
                select(Objection)
                .where(Objection.application_id == app.id, Objection.status == "ACIK")
                .order_by(Objection.created_at.desc())
            )
            .scalars()
            .first()
        )
        if objection is None:
            raise WorkbenchError("açık itiraz bulunamadı")
        objection.status = "KABUL" if upheld else "RED"
        objection.resolved_by = self.user.username
        objection.resolution = note
        objection.resolved_at = utcnow()
        pii = self.service.pii(app)
        if upheld:
            objection.letter = (
                f"Sayın {pii['name']},\n\n{app.id} numaralı başvurunuza ilişkin itirazınız kabul edilmiş ve "
                "başvurunuz bir kredi uzmanı tarafından yeniden değerlendirilmek üzere incelemeye alınmıştır. "
                "Sonuç ayrıca bildirilecektir.\n\nSaygılarımızla,\nBireysel Krediler Tahsis Birimi"
            )
            self.service.transition(app, State.UZMAN_INCELEMESI, "itiraz kabul: insan incelemesi")
        else:
            objection.letter = (
                f"Sayın {pii['name']},\n\n{app.id} numaralı başvurunuza ilişkin itirazınız bir kredi uzmanı "
                f"tarafından incelenmiştir. İnceleme sonucunda ilk karar korunmuştur. Gerekçe: {note}\n\n"
                "Kişisel verilerinize ilişkin haklarınız için KVKK m.11 kapsamında veri sorumlusuna başvurabilir, "
                "Kişisel Verileri Koruma Kurulu'na şikâyette bulunabilirsiniz.\n\n"
                "Saygılarımızla,\nBireysel Krediler Tahsis Birimi"
            )
            self.service.transition(app, State.REDDEDILDI, "itiraz reddedildi (insan incelemesi)")
        app.letters = {**(app.letters or {}), "itiraz_sonucu": objection.letter}
        self.service.audit(
            "OBJECTION_RESOLVED", app.id, {"objection_id": objection.id, "upheld": upheld}
        )
        return objection
