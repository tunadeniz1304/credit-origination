"""API router: liveness, application submission and status retrieval."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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


@router.get("/api/v1/applications/{application_id}/schedule", tags=["applications"])
def application_schedule(application_id: str) -> dict:
    """Return the amortization (repayment) plan for an approved application."""
    from app.engine.schedule import build_schedule

    settings = get_settings()
    record = get_record(application_id)
    result = load_result(application_id, settings)
    if result is None and record is None:
        raise HTTPException(status_code=404, detail="application not found")
    if result is None:
        result = record.result
    if result is None or result.decision is None:
        raise HTTPException(status_code=404, detail="no decision available yet")
    if result.status != ApplicationStatus.APPROVED:
        raise HTTPException(status_code=409, detail="schedule available only for approved applications")
    schedule = build_schedule(
        application_id,
        principal=result.decision.suggested_amount,
        term_months=result.decision.suggested_term_months,
    )
    return schedule.model_dump(mode="json")


@router.post("/api/v1/applications/{application_id}/documents", tags=["applications"])
async def upload_document(
    application_id: str,
    code: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Deliver a missing document; persists file + updates document status."""
    from pathlib import Path

    from app.agents.document_agent import DocumentControlAgent
    from app.engine.store import append_submitted_document

    settings = get_settings()
    normalized = code.upper()
    known = {
        req.code
        for req in load_pipeline_rules().document_policy.required_documents
    }
    if normalized not in known:
        raise HTTPException(status_code=400, detail=f"unknown document code: {code}")

    uploads = settings.uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename).suffix or ".txt"
    dest = uploads / f"{normalized}{extension}"
    dest.write_bytes(await file.read())

    updated = append_submitted_document(application_id, normalized, settings)
    if updated is None:
        raise HTTPException(status_code=404, detail="application not found")

    check = DocumentControlAgent().check(updated)
    return {
        "application_id": application_id,
        "uploaded_code": normalized,
        "path": str(dest),
        "complete": check.complete,
        "present": check.present,
        "missing": [req.code for req in check.missing],
    }


@router.post("/api/v1/applications/{application_id}/reprocess", tags=["applications"])
def reprocess_application(application_id: str) -> dict:
    """Re-enqueue an application (e.g. after delivering missing documents)."""
    settings = get_settings()
    from app.engine.store import load_application

    if load_application(application_id, settings) is None:
        raise HTTPException(status_code=404, detail="application not found")
    update_record_status(application_id, status=ApplicationStatus.QUEUED)
    receipt = TaskDispatcher().enqueue(
        "app.tasks.process_application", application_id=application_id
    )
    if receipt.backend == "celery":
        update_record_status(application_id, status=ApplicationStatus.PROCESSING)
    record = get_record(application_id)
    if record is not None:
        record.task_id = receipt.task_id
        record.queue_backend = receipt.backend
        store_record(record)
    return record.to_dict() if record else receipt.model_dump(mode="json")
