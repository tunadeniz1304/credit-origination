"""AI underwriter memo and policy assistant endpoints (staff only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.policy_assistant import ask
from app.agents.underwriter.graph import UnderwriterAgent
from app.api.deps import db_session, load_application, require_staff
from app.api.schemas import PolicyQuestion
from app.core.config import get_settings
from app.core.security import Principal
from app.core.task_dispatcher import run_coroutine_safe
from app.reports.memo_pdf import build_memo_pdf

router = APIRouter(tags=["agent"])


@router.post("/api/v1/agent/{application_id}/memo", status_code=status.HTTP_201_CREATED)
def generate_memo(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    agent = UnderwriterAgent(session, app, actor=f"ajan:{user.username}")
    memo = run_coroutine_safe(agent.run())
    settings = get_settings()
    pdf_path = settings.report_dir / app.id / f"{app.id}_ai_memorandum.pdf"
    build_memo_pdf(memo, pdf_path)
    app.letters = {**(app.letters or {}), "ai_memo": memo.model_dump()}
    app.reports = {**(app.reports or {}), "memo_pdf": str(pdf_path)}
    return {
        **memo.model_dump(),
        "pdf_available": True,
        "disclaimer": "Bu memorandum öneri niteliğindedir; bağlayıcı karar yetkili kredi personeline aittir.",
    }


@router.get("/api/v1/agent/{application_id}/memo")
def get_memo(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict[str, Any]:
    app = load_application(session, application_id, user)
    memo = (app.letters or {}).get("ai_memo")
    if not memo:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "memorandum henüz üretilmedi")
    return memo


@router.post("/api/v1/policy/ask")
def policy_ask(body: PolicyQuestion, user: Principal = Depends(require_staff)) -> dict[str, Any]:
    return run_coroutine_safe(ask(body.question))
