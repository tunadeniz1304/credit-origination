"""API router: liveness plus (Step-4 onwards) application endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import load_pipeline_rules
from app.core.task_dispatcher import TaskDispatcher

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict:
    """Liveness probe: service, queue backend and rules file status."""
    dispatcher = TaskDispatcher()
    rules = load_pipeline_rules()
    required = [req.code for req in rules.document_policy.required_documents]
    return {
        "status": "ok",
        "service": "credit_operations_agent",
        "queue_backend": dispatcher.backend,
        "required_documents": required,
    }


@router.post("/api/v1/ping", tags=["system"])
def ping() -> dict:
    """Exercise the task pipeline end-to-end (queue + task execution)."""
    dispatch = TaskDispatcher()
    receipt = dispatch.enqueue("app.tasks.ping")
    return receipt.model_dump(mode="json")
