"""Pricing simulation and decision replay."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import current_user, db_session, require_roles
from app.api.schemas import QuoteRequest
from app.core.rules import load_policy_file
from app.core.security import Principal
from app.db.models import Decision
from app.pricing.engine import quote
from app.workflow.pipeline import replay_decision

router = APIRouter(tags=["pricing"])

BAND_PD = {"A": 0.015, "B": 0.04, "C": 0.08, "D": 0.15}


@router.post("/api/v1/pricing/quote")
def pricing_quote(body: QuoteRequest, user: Principal = Depends(current_user)) -> dict[str, Any]:
    """Indicative quote. Applicants simulate by risk band; staff may pass a PD."""
    pd = body.pd if (body.pd is not None and user.is_staff) else BAND_PD[body.risk_band or "B"]
    policy = load_policy_file().product(body.product)
    if not policy.min_term <= body.term_months <= policy.max_term:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"vade {policy.min_term}-{policy.max_term} ay aralığında olmalı",
        )
    result = quote(pd=pd, amount=body.amount, term_months=body.term_months, product=body.product)
    return {
        **result.model_dump(),
        "indicative": True,
        "pd_source": "girdi" if body.pd and user.is_staff else f"bant {body.risk_band or 'B'}",
    }


@router.post("/api/v1/decisions/{decision_id}/replay")
def replay(
    decision_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(
        require_roles("uzman", "kidemli_uzman", "komite", "model_yoneticisi")
    ),
) -> dict[str, Any]:
    decision = session.get(Decision, decision_id)
    if decision is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "karar bulunamadı")
    if decision.kind != "engine":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "yalnızca motor kararları yeniden üretilebilir"
        )
    return replay_decision(session, decision)
