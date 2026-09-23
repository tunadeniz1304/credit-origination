"""API router: liveness, application submission and status retrieval."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import ApplicationSubmit
from app.api.store import get_record, list_records, store_record, update_record_status
from app.core.config import get_settings, load_pipeline_rules
from app.core.logging import get_logger
from app.core.task_dispatcher import TaskDispatcher
from app.engine.store import generate_application_id, load_result, persist_application
from app.models import Applicant, ApplicationRecord, ApplicationStatus, LoanApplication

router = APIRouter()
logger = get_logger("api.router")


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
    receipt = TaskDispatcher().enqueue("app.tasks.ping")
    return receipt.model_dump(mode="json")


@router.post("/api/v1/applications", tags=["applications"], status_code=202)
def submit_application(payload: ApplicationSubmit) -> dict:
    """Accept a credit application, persist it and enqueue processing."""
    application = LoanApplication(
        applicant=Applicant(
            name=payload.name,
            identity_no=payload.identity_no,
            monthly_income=payload.monthly_income,
            submitted_documents=payload.submitted_documents,
        ),
        requested_amount=payload.requested_amount,
        requested_term_months=payload.requested_term_months,
        currency=payload.currency,
    )
    settings = get_settings()
    application_id = generate_application_id()
    persist_application(application_id, application, settings)

    record = ApplicationRecord(
        application_id=application_id,
        application=application,
        status=ApplicationStatus.QUEUED,
    )
    store_record(record)

    try:
        receipt = TaskDispatcher().enqueue(
            "app.tasks.process_application", application_id=application_id
        )
    except Exception as exc:  # noqa: BLE001 - dispatch failure must not 500 the POST
        logger.exception("Dispatch failed for %s", application_id)
        update_record_status(
            application_id,
            status=ApplicationStatus.FAILED,
            error=f"dispatch failed: {exc}",
        )
        return get_record(application_id).to_dict()

    if receipt.backend == "celery":
        update_record_status(application_id, status=ApplicationStatus.PROCESSING)

    record = get_record(application_id)
    record.task_id = receipt.task_id
    record.queue_backend = receipt.backend
    store_record(record)
    return record.to_dict()


@router.get("/api/v1/applications/{application_id}", tags=["applications"])
def get_application(application_id: str) -> dict:
    """Return an application's status/results (persisted store first)."""
    settings = get_settings()
    record = get_record(application_id)
    result = load_result(application_id, settings)
    if record is None and result is None:
        raise HTTPException(status_code=404, detail="application not found")
    if record is not None:
        if result is not None:
            record.result = result
            if result.status in (ApplicationStatus.APPROVED, ApplicationStatus.REJECTED):
                record.status = result.status
            store_record(record)
        return record.to_dict()
    return {
        "application_id": application_id,
        "application": result.application.model_dump(mode="json"),
        "status": result.status.value,
        "task_id": None,
        "queue_backend": "persisted",
        "result": result.model_dump(mode="json"),
        "error": None,
    }


@router.get("/api/v1/applications", tags=["applications"])
def list_applications(limit: int = 50) -> dict:
    """Return recent applications, newest first."""
    return {"applications": list_records(limit=min(max(limit, 1), 200))}
