"""System endpoints: health probes, metrics, queue, LLM status, audit, outbox."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.agents.llm_service import llm_status
from app.api.deps import current_user, db_session, require_roles, require_staff
from app.core.config import get_settings, load_pipeline_rules
from app.core.metrics import QUEUE_DEPTH, render_latest
from app.core.security import Principal
from app.db.audit import verify_chain
from app.db.models import Application, Decision, LLMCall, OutboxMessage
from app.db.outbox import dispatch_pending, mark_delivered
from app.integrations.circuit_breaker import all_breakers
from app.workflow.dispatch import active_backend
from app.workflow.states import STATE_LABELS, State

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, Any]:
    """Backwards-compatible liveness summary."""
    rules = load_pipeline_rules()
    return {
        "status": "ok",
        "service": "anil2_credit_platform",
        "queue_backend": active_backend(),
        "required_documents": [d.code for d in rules.document_policy.required_documents],
        "llm_mode": get_settings().llm_effective_mode,
    }


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready")
def ready(response: Response, session: Session = Depends(db_session)) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # pragma: no cover - database down
        checks["database"] = f"error: {type(exc).__name__}"
    backend = active_backend()
    if backend == "celery":
        try:
            import redis

            redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=0.5).ping()
            checks["redis"] = "ok"
        except Exception as exc:  # pragma: no cover
            checks["redis"] = f"error: {type(exc).__name__}"
    else:
        checks["redis"] = "not required (inline)"
    try:
        from app.decisioning.models import get_models

        checks["models"] = get_models().versions
    except Exception as exc:  # pragma: no cover
        checks["models"] = f"error: {type(exc).__name__}"
    ok = (
        checks["database"] == "ok"
        and isinstance(checks["models"], dict)
        and checks["redis"] != "error"
    )
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "degraded", "checks": checks}


@router.get("/metrics", include_in_schema=False)
def prometheus(session: Session = Depends(db_session)) -> Response:
    QUEUE_DEPTH.set(
        session.execute(
            select(func.count(Application.id)).where(
                Application.state == State.UZMAN_INCELEMESI.value
            )
        ).scalar_one()
    )
    return Response(render_latest(), media_type="text/plain; version=0.0.4")


@router.get("/api/v1/metrics")
def summary_metrics(
    session: Session = Depends(db_session), user: Principal = Depends(require_staff)
) -> dict[str, Any]:
    """Operational summary computed from the database (not process memory)."""
    by_state = dict(
        session.execute(
            select(Application.state, func.count(Application.id)).group_by(Application.state)
        ).all()
    )
    outcomes = dict(
        session.execute(
            select(Decision.outcome, func.count(Decision.id))
            .where(Decision.kind == "engine")
            .group_by(Decision.outcome)
        ).all()
    )
    engine_total = sum(outcomes.values())
    automated = outcomes.get("OTOMATIK_ONAY", 0) + outcomes.get("OTOMATIK_RET", 0)
    approved_states = {
        State.OTOMATIK_ONAY.value,
        State.TEKLIF_SUNULDU.value,
        State.TEKLIF_KABUL.value,
        State.SOZLESME_HAZIR.value,
        State.KULLANDIRILDI.value,
    }
    decided = [
        s
        for s in by_state
        if s in approved_states | {State.OTOMATIK_RET.value, State.REDDEDILDI.value}
    ]
    approved = sum(by_state.get(s, 0) for s in approved_states)
    decided_total = sum(by_state.get(s, 0) for s in decided)
    pds = [
        p
        for (p,) in session.execute(
            select(Decision.pd).where(Decision.kind == "engine", Decision.pd.is_not(None))
        ).all()
    ]
    latency = session.execute(select(func.avg(Decision.latency_ms))).scalar_one()
    llm = dict(
        session.execute(select(LLMCall.mode, func.count(LLMCall.id)).group_by(LLMCall.mode)).all()
    )
    buckets = [0.02, 0.05, 0.1, 0.2, 1.0]
    distribution = {
        f"<= {b:.0%}": sum(1 for p in pds if p <= b and (i == 0 or p > buckets[i - 1]))
        for i, b in enumerate(buckets)
    }
    return {
        "total_applications": sum(by_state.values()),
        "by_state": by_state,
        "by_state_labels": {STATE_LABELS[State(k)]: v for k, v in by_state.items()},
        "decisions": outcomes,
        "automation_rate": round(automated / engine_total, 4) if engine_total else None,
        "approval_rate": round(approved / decided_total, 4) if decided_total else None,
        "average_decision_latency_ms": round(latency, 1) if latency else None,
        "pd_distribution": distribution,
        "review_queue_depth": by_state.get(State.UZMAN_INCELEMESI.value, 0),
        "llm_calls_by_mode": llm,
        "circuit_breakers": all_breakers(),
    }


@router.get("/api/v1/queue")
def queue_status(user: Principal = Depends(require_staff)) -> dict[str, Any]:
    from app.worker.tasks import INLINE_TASKS

    backend = active_backend()
    body: dict[str, Any] = {
        "backend": backend,
        "registered_tasks": sorted(INLINE_TASKS),
        "celery_active": None,
    }
    if backend == "celery":  # pragma: no cover - requires a broker
        try:
            from app.worker.tasks import _celery

            active = _celery.control.inspect(timeout=1).active() or {}
            body["celery_active"] = {worker: len(tasks) for worker, tasks in active.items()}
        except Exception as exc:
            body["celery_active_error"] = type(exc).__name__
    return body


@router.get("/api/v1/llm/status")
def llm_status_endpoint(user: Principal = Depends(current_user)) -> dict[str, Any]:
    return llm_status()


@router.get("/api/v1/audit/verify")
def audit_verify(
    session: Session = Depends(db_session),
    user: Principal = Depends(require_roles("komite", "model_yoneticisi")),
) -> dict[str, Any]:
    return verify_chain(session)


@router.get("/api/v1/notifications")
def notifications(
    limit: int = 50,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    rows = (
        session.execute(
            select(OutboxMessage)
            .order_by(OutboxMessage.status.desc(), OutboxMessage.created_at.desc())
            .limit(max(1, min(limit, 200)))
        )
        .scalars()
        .all()
    )
    return {
        "total": len(rows),
        "pending": sum(1 for r in rows if r.status == "PENDING"),
        "entries": [
            {
                "id": r.id,
                "event": r.event,
                "aggregate_id": r.aggregate_id,
                "channel": r.channel,
                "status": r.status,
                "attempts": r.attempts,
                "created_at": r.created_at.isoformat(),
                "delivered": r.status == "SENT",
            }
            for r in rows
        ],
    }


@router.post("/api/v1/notifications/dispatch")
def notifications_dispatch(user: Principal = Depends(require_roles())) -> dict[str, Any]:
    return dispatch_pending()


@router.post("/api/v1/notifications/{message_id}/deliver")
def notification_deliver(
    message_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_roles()),
) -> dict[str, Any]:
    if not mark_delivered(session, message_id):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "bildirim bulunamadı veya zaten teslim edildi"
        )
    return {"notification_id": message_id, "delivered": True}
