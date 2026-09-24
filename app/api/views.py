"""Role-aware JSON projections of ORM objects.

Applicants see their own data and plain-language outputs; staff see the full
decision detail. TCKN / IBAN / phone are always masked in API responses.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import crypto
from app.core.security import Principal
from app.db.models import Application, CashflowFeatures, Decision, Document, ExtractedField, Offer
from app.decisioning.features import FEATURE_LABELS
from app.workflow.states import STATE_LABELS, State


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def applicant_view(app: Application) -> dict[str, Any]:
    a = app.applicant
    name = crypto.decrypt(a.name_enc) or ("(anonimleştirildi)" if a.anonymized_at else "")
    return {
        "name": name,
        "identity_no_masked": crypto.mask_tckn(crypto.decrypt(a.tckn_enc)),
        "phone_masked": crypto.mask_phone(crypto.decrypt(a.phone_enc)),
        "iban_masked": crypto.mask_iban(crypto.decrypt(a.iban_enc)),
        "email": crypto.decrypt(a.email_enc),
        "birth_date": _iso(a.birth_date),
        "anonymized": a.anonymized_at is not None,
    }


def application_summary(app: Application) -> dict[str, Any]:
    return {
        "application_id": app.id,
        "state": app.state,
        "state_label": STATE_LABELS.get(State(app.state), app.state),
        "product": app.product,
        "requested_amount": app.requested_amount,
        "requested_term_months": app.requested_term_months,
        "currency": app.currency,
        "applicant_name": applicant_view(app)["name"],
        "created_at": _iso(app.created_at),
        "updated_at": _iso(app.updated_at),
        "sla_due_at": _iso(app.sla_due_at),
        "assigned_to": app.assigned_to,
        "priority": app.priority,
        "persona": app.persona,
    }


def decision_view(decision: Decision, user: Principal) -> dict[str, Any]:
    public = {
        "decision_id": decision.id,
        "kind": decision.kind,
        "outcome": decision.outcome,
        "outcome_label": STATE_LABELS.get(State(decision.outcome), decision.outcome)
        if decision.outcome in State.__members__
        else decision.outcome,
        "conditional": decision.conditional,
        "reason_codes": [{"code": r["code"], "text": r["text"]} for r in decision.reason_codes],
        "counterfactuals": [c["text"] for c in decision.counterfactuals],
        "created_at": _iso(decision.created_at),
        "applicant_letter": (decision.narratives or {}).get("applicant_letter"),
        "objection_right": decision.outcome in ("OTOMATIK_RET", "REDDEDILDI"),
    }
    if not user.is_staff and user.role != "model_yoneticisi":
        return public
    explanation = decision.explanation or {}
    shap = explanation.get("shap", {})
    return {
        **public,
        "reason_codes": decision.reason_codes,
        "pd": decision.pd,
        "pd_requested": explanation.get("pd_requested"),
        "risk_band": decision.risk_band,
        "score_points": decision.score_points,
        "scorecard": explanation.get("scorecard"),
        "rule_results": decision.rule_results,
        "limits": decision.limits,
        "pricing": {k: v for k, v in (decision.pricing or {}).items() if k != "schedule"},
        "counterfactuals_detail": decision.counterfactuals,
        "shap": [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f),
                "value": v,
                "input": (decision.feature_snapshot or {}).get(f),
            }
            for f, v in shap.items()
        ],
        "shap_base_value": explanation.get("base_value"),
        "challenger": decision.challenger,
        "rule_set_version": decision.rule_set_version,
        "model_version": decision.model_version,
        "feature_hash": decision.feature_hash,
        "feature_snapshot": decision.feature_snapshot,
        "narratives": decision.narratives,
        "latency_ms": decision.latency_ms,
        "decided_by": decision.decided_by,
    }


def offer_view(offer: Offer, include_schedule: bool = True) -> dict[str, Any]:
    body = {
        "offer_id": offer.id,
        "amount": offer.amount,
        "term_months": offer.term_months,
        "annual_rate": offer.annual_rate,
        "instalment": offer.instalment,
        "apr": offer.apr,
        "total_payment": offer.total_payment,
        "fees": offer.fees,
        "valid_until": _iso(offer.valid_until),
        "status": offer.status,
        "accepted_at": _iso(offer.accepted_at),
        "total_taxes": (offer.details or {}).get("total_taxes"),
    }
    if include_schedule:
        body["schedule"] = (offer.details or {}).get("schedule", [])
    return body


def document_view(session: Session, document: Document, user: Principal) -> dict[str, Any]:
    body: dict[str, Any] = {
        "document_id": document.id,
        "code": document.code,
        "filename": document.filename,
        "mime": document.mime,
        "size": document.size,
        "status": document.status,
        "sha256": document.sha256,
        "uploaded_at": _iso(document.created_at),
        "text_source": document.text_source,
    }
    if user.is_staff:
        fields = session.execute(
            select(ExtractedField)
            .where(ExtractedField.document_id == document.id)
            .order_by(ExtractedField.id)
        ).scalars()
        body["fraud_score"] = document.fraud_score
        body["fraud_signals"] = document.fraud_signals
        body["fields"] = [
            {
                "field_id": f.id,
                "name": f.name,
                "value": f.value,
                "confidence": f.confidence,
                "page": f.page,
                "bbox": f.bbox,
                "source": f.source,
                "corrected_by": f.corrected_by,
            }
            for f in fields
        ]
    return body


def cashflow_view(row: CashflowFeatures | None) -> dict[str, Any]:
    if row is None:
        return {"available": False, "features": {}, "monthly": [], "categories": {}}
    return {
        "available": True,
        "features": row.features,
        "monthly": row.monthly,
        "categories": row.categories,
    }


def timeline_view(app: Application) -> list[dict[str, Any]]:
    return [
        {
            "from": e.from_state,
            "to": e.to_state,
            "to_label": STATE_LABELS.get(State(e.to_state), e.to_state),
            "actor": e.actor,
            "reason": e.reason,
            "at": _iso(e.created_at),
        }
        for e in app.events
    ]
