"""Underwriter workbench endpoints (staff only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_session, load_application, require_roles, require_staff
from app.api.schemas import (
    CheckerRequest,
    FieldCorrection,
    ObjectionResolution,
    ReviewDecisionRequest,
)
from app.core.security import Principal
from app.db.models import ExtractedField, Objection, Review
from app.workbench.authority import evaluate_authority
from app.workbench.service import Workbench, WorkbenchError, queue
from app.workflow.offers import disburse
from app.workflow.pipeline import latest_decision
from app.workflow.states import InvalidTransitionError, State

router = APIRouter(prefix="/api/v1/workbench", tags=["workbench"])


def _raise(exc: Exception) -> None:
    if isinstance(exc, WorkbenchError):
        raise HTTPException(exc.status, str(exc)) from exc
    raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _review_view(review: Review) -> dict[str, Any]:
    return {
        "review_id": review.id,
        "application_id": review.application_id,
        "maker": review.maker,
        "maker_role": review.maker_role,
        "action": review.action,
        "amount": review.amount,
        "term_months": review.term_months,
        "annual_rate": review.annual_rate,
        "justification": review.justification,
        "is_override": review.is_override,
        "required_role": review.required_role,
        "four_eyes": review.four_eyes,
        "status": review.status,
        "checker": review.checker,
        "checker_note": review.checker_note,
        "created_at": review.created_at.isoformat(),
    }


@router.get("/queue")
def review_queue(
    state: str | None = Query(default=None, pattern="^(UZMAN_INCELEMESI|ITIRAZ_INCELEMESI)$"),
    mine: bool = False,
    q: str | None = Query(default=None, max_length=64),
    product: str | None = Query(default=None, pattern="^[A-Z_]{2,20}$"),
    sla_breached: bool | None = None,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    page = queue(
        session,
        state=state,
        assigned_to=user.username if mine else None,
        search=q,
        product=product,
        sla_breached=sla_breached,
        limit=limit,
        offset=offset,
    )
    return {**page, "count": len(page["items"])}


@router.post("/{application_id}/assign")
def assign(
    application_id: str,
    assignee: str | None = None,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    try:
        Workbench(session, user).assign(app, assignee)
    except WorkbenchError as exc:
        _raise(exc)
    return {"application_id": app.id, "assigned_to": app.assigned_to}


@router.get("/{application_id}/authority")
def authority_preview(
    application_id: str,
    amount: float | None = None,
    override: bool = False,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    decision = latest_decision(session, app.id)
    pd = decision.pd if decision and decision.pd is not None else 0.1
    check = evaluate_authority(
        user, amount=amount or app.requested_amount, pd=pd, is_override=override
    )
    return check.model_dump()


@router.post("/{application_id}/decision", status_code=status.HTTP_201_CREATED)
def submit_decision(
    application_id: str,
    body: ReviewDecisionRequest,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    try:
        review = Workbench(session, user).submit_decision(
            app,
            action=body.action,
            justification=body.justification,
            amount=body.amount,
            term_months=body.term_months,
            annual_rate=body.annual_rate,
        )
    except (WorkbenchError, InvalidTransitionError) as exc:
        _raise(exc)
    return {"application_id": app.id, "state": app.state, "review": _review_view(review)}


@router.get("/reviews")
def reviews(
    status_filter: str = Query(default="ONAY_BEKLIYOR", alias="status"),
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    rows = (
        session.execute(
            select(Review).where(Review.status == status_filter).order_by(Review.created_at.desc())
        )
        .scalars()
        .all()
    )
    return {"reviews": [_review_view(r) for r in rows]}


@router.post("/reviews/{review_id}/check")
def check_review(
    review_id: str,
    body: CheckerRequest,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    review = session.get(Review, review_id)
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "karar talebi bulunamadı")
    try:
        Workbench(session, user).check(review, approve=body.approve, note=body.note)
    except (WorkbenchError, InvalidTransitionError) as exc:
        _raise(exc)
    app_state = load_application(session, review.application_id, user).state
    return {"review": _review_view(review), "state": app_state}


@router.post("/{application_id}/fields/{field_id}")
def correct_field(
    application_id: str,
    field_id: int,
    body: FieldCorrection,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    field = session.get(ExtractedField, field_id)
    if field is None or field.application_id != app.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "alan bulunamadı")
    old = field.value
    field.value, field.confidence, field.source, field.corrected_by = (
        body.value,
        1.0,
        "manual",
        user.username,
    )
    Workbench(session, user).service.audit(
        "FIELD_CORRECTED",
        app.id,
        {
            "field_id": field_id,
            "name": field.name,
            "old": old,
            "new": body.value,
            "note": body.note,
        },
    )
    return {
        "field_id": field.id,
        "name": field.name,
        "value": field.value,
        "confidence": field.confidence,
        "corrected_by": field.corrected_by,
    }


@router.get("/objections")
def objections(
    session: Session = Depends(db_session), user: Principal = Depends(require_staff)
) -> dict[str, Any]:
    rows = session.execute(select(Objection).order_by(Objection.created_at.desc())).scalars().all()
    return {
        "objections": [
            {
                "objection_id": o.id,
                "application_id": o.application_id,
                "reason": o.reason,
                "status": o.status,
                "due_at": o.due_at.isoformat(),
                "resolved_by": o.resolved_by,
                "resolution": o.resolution,
            }
            for o in rows
        ]
    }


@router.post("/{application_id}/objection/resolve")
def resolve_objection(
    application_id: str,
    body: ObjectionResolution,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    try:
        objection = Workbench(session, user).resolve_objection(
            app, upheld=body.upheld, note=body.note
        )
    except (WorkbenchError, InvalidTransitionError) as exc:
        _raise(exc)
    return {
        "application_id": app.id,
        "state": app.state,
        "objection_status": objection.status,
        "letter": objection.letter,
    }


@router.post("/{application_id}/disburse")
def disburse_loan(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_roles("uzman", "kidemli_uzman", "komite")),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    if app.state != State.SOZLESME_HAZIR.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "kullandırım için sözleşme hazır olmalı")
    try:
        disburse(session, app, user)
    except InvalidTransitionError as exc:
        _raise(exc)
    return {"application_id": app.id, "state": app.state}
