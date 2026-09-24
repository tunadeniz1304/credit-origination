"""Model governance, fairness, drift, rule sets and portfolio early warning."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_roles
from app.api.schemas import RuleSetDraft
from app.core.config import get_settings
from app.core.security import Principal
from app.db.models import Applicant, Application, Decision, LoanPerformance, ModelRecord, RuleSet
from app.ews.model import watchlist_entry
from app.governance.drift import compute_drift_report
from app.governance.fairness import live_fairness
from app.governance.inventory import (
    GovernanceError,
    approve_promotion,
    approve_rule_set,
    backtest_rule_set,
    card_markdown,
    card_pdf,
    champion_challenger,
    model_card,
    submit_rule_set,
    sync_inventory,
)
from app.workflow.service import age_on
from app.workflow.states import State

router = APIRouter(tags=["governance"])
MODEL_ROLES = require_roles("model_yoneticisi", "komite")
STAFF_OR_MODEL = require_roles("uzman", "kidemli_uzman", "komite", "model_yoneticisi")


def _err(exc: GovernanceError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@router.get("/api/v1/models")
def models(
    session: Session = Depends(db_session), user: Principal = Depends(STAFF_OR_MODEL)
) -> dict[str, Any]:
    rows = sync_inventory(session)
    return {
        "models": [
            {
                "model_id": r.id,
                "name": r.name,
                "kind": r.kind,
                "role": r.role,
                "status": r.status,
                "metrics": {k: v for k, v in (r.metrics or {}).items() if k != "calibration"},
                "approvals": r.approvals,
            }
            for r in rows
        ]
    }


@router.get("/api/v1/models/{model_id}/card")
def card(
    model_id: str,
    format: str = Query(default="json", pattern="^(json|md|pdf)$"),
    session: Session = Depends(db_session),
    user: Principal = Depends(STAFF_OR_MODEL),
) -> Any:
    sync_inventory(session)
    row = session.get(ModelRecord, model_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model bulunamadı")
    data = model_card(row)
    if format == "md":
        return PlainTextResponse(card_markdown(data), media_type="text/markdown; charset=utf-8")
    if format == "pdf":
        path = card_pdf(data, get_settings().report_dir / "models" / f"{model_id}_model_karti.pdf")
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    return data


@router.get("/api/v1/governance/champion-challenger")
def cc(
    session: Session = Depends(db_session), user: Principal = Depends(STAFF_OR_MODEL)
) -> dict[str, Any]:
    return champion_challenger(session)


@router.post("/api/v1/models/{model_id}/promote")
def promote(
    model_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_roles("model_yoneticisi")),
) -> dict[str, Any]:
    sync_inventory(session)
    try:
        row = approve_promotion(session, model_id, user.username)
    except GovernanceError as exc:
        raise _err(exc) from exc
    return {"model_id": row.id, "role": row.role, "status": row.status, "approvals": row.approvals}


@router.get("/api/v1/governance/drift")
def drift(
    session: Session = Depends(db_session), user: Principal = Depends(STAFF_OR_MODEL)
) -> dict[str, Any]:
    return compute_drift_report(session)


@router.get("/api/v1/governance/fairness")
def fairness(
    session: Session = Depends(db_session), user: Principal = Depends(STAFF_OR_MODEL)
) -> dict[str, Any]:
    import json

    rows = []
    q = (
        select(Decision.outcome, Applicant.gender, Applicant.birth_date, Applicant.province)
        .join(Application, Application.id == Decision.application_id)
        .join(Applicant, Applicant.id == Application.applicant_id)
        .where(Decision.kind == "engine")
    )
    for outcome, gender, birth, province in session.execute(q).all():
        age = age_on(birth) if birth else None
        band = (
            None
            if age is None
            else "21-29"
            if age < 30
            else "30-39"
            if age < 40
            else "40-49"
            if age < 50
            else "50+"
        )
        rows.append(
            {
                "approved": outcome == "OTOMATIK_ONAY",
                "gender": gender,
                "age_band": band,
                "province": province,
            }
        )
    path = get_settings().models_path / "fairness.json"
    offline = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return {"live": live_fairness(rows), "offline": offline}


@router.get("/api/v1/rule-sets")
def rule_sets(
    session: Session = Depends(db_session), user: Principal = Depends(STAFF_OR_MODEL)
) -> dict[str, Any]:
    rows = session.execute(select(RuleSet).order_by(RuleSet.created_at.desc())).scalars().all()
    return {
        "active_file": "policy_v1",
        "rule_sets": [
            {"version": r.id, "status": r.status, "backtest": r.backtest, "approvals": r.approvals}
            for r in rows
        ],
    }


@router.post("/api/v1/rule-sets/backtest")
def backtest(
    body: RuleSetDraft,
    session: Session = Depends(db_session),
    user: Principal = Depends(MODEL_ROLES),
) -> dict[str, Any]:
    try:
        return backtest_rule_set(session, body.content)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"geçersiz kural seti: {type(exc).__name__}"
        ) from exc


@router.post("/api/v1/rule-sets", status_code=status.HTTP_201_CREATED)
def submit(
    body: RuleSetDraft,
    session: Session = Depends(db_session),
    user: Principal = Depends(MODEL_ROLES),
) -> dict[str, Any]:
    try:
        row = submit_rule_set(session, body.version, body.content, user.username)
    except GovernanceError as exc:
        raise _err(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"geçersiz kural seti: {type(exc).__name__}"
        ) from exc
    return {"version": row.id, "status": row.status, "backtest": row.backtest}


@router.post("/api/v1/rule-sets/{version}/approve")
def approve(
    version: str, session: Session = Depends(db_session), user: Principal = Depends(MODEL_ROLES)
) -> dict[str, Any]:
    try:
        row = approve_rule_set(session, version, user.username)
    except GovernanceError as exc:
        raise _err(exc) from exc
    return {"version": row.id, "status": row.status, "approvals": row.approvals}


@router.get("/api/v1/portfolio/watchlist")
def watchlist(
    session: Session = Depends(db_session), user: Principal = Depends(STAFF_OR_MODEL)
) -> dict[str, Any]:
    apps = (
        session.execute(select(Application).where(Application.state == State.KULLANDIRILDI.value))
        .scalars()
        .all()
    )
    entries = []
    for app in apps:
        history = [
            {
                "month": p.month,
                "dpd": p.dpd,
                "balance": p.balance,
                "salary_credited": p.salary_credited,
                "new_bureau_delinquency": p.new_bureau_delinquency,
                "utilisation": p.utilisation,
            }
            for p in session.execute(
                select(LoanPerformance)
                .where(LoanPerformance.application_id == app.id)
                .order_by(LoanPerformance.month)
            ).scalars()
        ]
        if not history:
            continue
        decision = (
            session.execute(
                select(Decision)
                .where(Decision.application_id == app.id)
                .order_by(Decision.created_at.desc())
            )
            .scalars()
            .first()
        )
        entries.append(
            watchlist_entry(app.id, history, decision.pd if decision and decision.pd else 0.05)
        )
    entries.sort(key=lambda e: e["ews_score"], reverse=True)
    return {
        "loans": len(entries),
        "on_watchlist": sum(1 for e in entries if e["on_watchlist"]),
        "entries": entries,
    }
