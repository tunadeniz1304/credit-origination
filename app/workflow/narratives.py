"""Build the citable narrative context from a persisted decision."""

from __future__ import annotations

from typing import Any

from app.agents.narrator import Fact, NarrativeContext, ReasonText
from app.core.rules import load_workflow
from app.db.models import Application, Decision


def decision_facts(app: Application, decision: Decision) -> list[Fact]:
    snap: dict[str, Any] = decision.feature_snapshot or {}
    limits: dict[str, Any] = decision.limits or {}
    pricing: dict[str, Any] = decision.pricing or {}
    facts = [
        Fact(
            id="f:income.monthly",
            label="Aylık net gelir",
            value=float(snap.get("monthly_income", app.declared_income)),
            unit="TL",
        ),
        Fact(
            id="f:income.declared",
            label="Beyan edilen gelir",
            value=float(app.declared_income),
            unit="TL",
        ),
        Fact(
            id="f:loan.amount",
            label="Talep edilen tutar",
            value=float(app.requested_amount),
            unit="TL",
        ),
        Fact(
            id="f:loan.term",
            label="Talep edilen vade",
            value=int(app.requested_term_months),
            unit="ay",
        ),
        Fact(
            id="f:dsr",
            label="Borç servis oranı (talep)",
            value=float(snap.get("dsr", 0.0)),
            unit="%",
        ),
        Fact(
            id="f:dsr.existing",
            label="Mevcut borç servis oranı",
            value=float(snap.get("existing_dsr", 0.0)),
            unit="%",
        ),
        Fact(
            id="f:policy.max_dsr",
            label="DSR politika sınırı",
            value=float(limits.get("max_dsr", 0.5)),
            unit="%",
        ),
        Fact(
            id="f:model.pd",
            label="12 aylık temerrüt olasılığı",
            value=float(decision.pd or 0.0),
            unit="%",
        ),
        Fact(
            id="f:scorecard.points",
            label="Skor kartı puanı",
            value=float(decision.score_points or 0.0),
            unit="puan",
        ),
        Fact(
            id="f:employment.months",
            label="Çalışma süresi",
            value=int(snap.get("employment_months", 0)),
            unit="ay",
        ),
        Fact(
            id="f:cashflow.income_cv",
            label="Gelir değişkenliği",
            value=float(snap.get("income_cv", 0.0)),
        ),
        Fact(
            id="f:cashflow.negative_days",
            label="Eksi bakiye gün",
            value=float(snap.get("negative_balance_days", 0.0)),
            unit="gün",
        ),
        Fact(
            id="f:bureau.inquiries",
            label="Son 6 ay sorgu",
            value=int(snap.get("inquiries_6m", 0)),
            unit="adet",
        ),
    ]
    if snap.get("bureau_score") is not None:
        facts.append(
            Fact(
                id="f:bureau.score",
                label="KKB kredi notu",
                value=int(snap["bureau_score"]),
                unit="puan",
            )
        )
    if limits.get("max_approvable_amount"):
        facts.append(
            Fact(
                id="f:limit.max_amount",
                label="Onaylanabilir azami tutar",
                value=float(limits["max_approvable_amount"]),
                unit="TL",
            )
        )
    if pricing:
        facts += [
            Fact(
                id="f:offer.amount",
                label="Teklif tutarı",
                value=float(pricing["amount"]),
                unit="TL",
            ),
            Fact(
                id="f:offer.term",
                label="Teklif vadesi",
                value=int(pricing["term_months"]),
                unit="ay",
            ),
            Fact(
                id="f:offer.instalment",
                label="Aylık taksit",
                value=float(pricing["instalment"]),
                unit="TL",
            ),
            Fact(
                id="f:pricing.annual_rate",
                label="Yıllık akdi faiz",
                value=float(pricing["annual_rate"]),
                unit="%",
            ),
            Fact(
                id="f:pricing.apr",
                label="Yıllık maliyet oranı",
                value=float(pricing["apr"]),
                unit="%",
            ),
            Fact(
                id="f:pricing.total_payment",
                label="Toplam geri ödeme",
                value=float(pricing["total_payment"]),
                unit="TL",
            ),
        ]
    return facts


def reason_facts(decision: Decision) -> list[Fact]:
    """Numbers quoted inside reason texts must be citable too (letters guard)."""
    snap = decision.feature_snapshot or {}
    extra: list[Fact] = []
    for key in (
        "gambling_share",
        "savings_rate",
        "bureau_utilisation",
        "loan_to_income",
        "delinquency_count_24m",
        "nsf_count",
    ):
        if key in snap and snap[key] is not None:
            extra.append(Fact(id=f"f:snap.{key}", label=key, value=float(snap[key])))
    return extra


def build_context(
    app: Application,
    decision: Decision,
    *,
    applicant_name: str,
    pii: dict[str, Any],
    outcome: str | None = None,
    fraud_flags: list[str] | None = None,
) -> NarrativeContext:
    workflow = load_workflow()
    return NarrativeContext(
        application_id=app.id,
        applicant_name=applicant_name,
        outcome=outcome or decision.outcome,  # type: ignore[arg-type]
        conditional=decision.conditional,
        facts=decision_facts(app, decision) + reason_facts(decision),
        reason_codes=[ReasonText(code=r["code"], text=r["text"]) for r in decision.reason_codes],
        counterfactuals=[c["text"] for c in decision.counterfactuals],
        fraud_flags=fraud_flags or [],
        objection_deadline_days=workflow.objection_window_days,
        review_sla_hours=workflow.sla_hours.get("UZMAN_INCELEMESI", 24),
        redactor_fields={k: v for k, v in pii.items() if v and k != "name"},
    )


def missing_docs_context(
    app: Application, applicant_name: str, missing: list[tuple[str, str]], pii: dict[str, Any]
) -> NarrativeContext:
    return NarrativeContext(
        application_id=app.id,
        applicant_name=applicant_name,
        outcome="BELGE_EKSIK",
        missing_documents=[desc for _, desc in missing],
        redactor_fields={k: v for k, v in pii.items() if v and k != "name"},
    )
