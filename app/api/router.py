"""API router: liveness, application submission and status retrieval."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

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
    from app.engine.audit import (
        ACTION_QUEUED,
        ACTION_SUBMITTED,
        append_audit,
        load_audit,
    )

    # Fraud-visibility: flag a repeat application by the same identity number.
    previous_id: str | None = None
    for rec in list_records(limit=200):
        if (
            rec["application"]["applicant"]["identity_no"] == payload.identity_no
            and rec["status"] != "FAILED"
        ):
            previous_id = rec["application_id"]
            break

    detail = "application accepted"
    if previous_id is not None:
        detail += f" (repeat identity; previous application {previous_id})"
    append_audit(application_id, ACTION_SUBMITTED, detail, settings)

    record = ApplicationRecord(
        application_id=application_id,
        application=application,
        status=ApplicationStatus.QUEUED,
    )
    store_record(record)

    append_audit(
        application_id,
        ACTION_QUEUED,
        f"queued on backend={settings.task_queue_backend}",
        settings,
    )

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
    from app.engine.audit import ACTION_DOCUMENT_DELIVERED, append_audit

    append_audit(
        application_id, ACTION_DOCUMENT_DELIVERED, f"{normalized} received", settings
    )

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
    from app.engine.audit import ACTION_REPROCESSED, append_audit

    append_audit(application_id, ACTION_REPROCESSED, "application re-enqueued", settings)
    if receipt.backend == "celery":
        update_record_status(application_id, status=ApplicationStatus.PROCESSING)
    record = get_record(application_id)
    if record is not None:
        record.task_id = receipt.task_id
        record.queue_backend = receipt.backend
        store_record(record)
    return record.to_dict() if record else receipt.model_dump(mode="json")


@router.get("/api/v1/applications/{application_id}/audit", tags=["applications"])
def application_audit(application_id: str) -> dict:
    """Return the append-only audit trail for an application."""
    from app.engine.audit import load_audit

    settings = get_settings()
    if get_record(application_id) is None and load_result(application_id, settings) is None:
        raise HTTPException(status_code=404, detail="application not found")
    entries = load_audit(application_id, settings)
    return {
        "application_id": application_id,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }


@router.get("/api/v1/metrics", tags=["system"])
def metrics() -> dict:
    """Summarize the processed pipeline (status distribution + totals)."""
    settings = get_settings()
    from app.api.store import records as _records

    status_counts: dict[str, int] = {}
    total_suggested = 0.0
    approved_count = 0
    for record in _records.values():
        status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
        if record.result and record.result.decision:
            total_suggested += record.result.decision.suggested_amount
            if record.result.status == ApplicationStatus.APPROVED:
                approved_count += 1
    return {
        "total_applications": len(_records),
        "by_status": status_counts,
        "approved_count": approved_count,
        "sum_suggested_amount": round(total_suggested, 2),
    }


@router.get("/api/v1/queue", tags=["system"])
def queue_status() -> dict:
    """Report the active queue backend, registered tasks and broker health."""
    dispatcher = TaskDispatcher()
    from app.worker.tasks import INLINE_TASKS

    registered = sorted(INLINE_TASKS.keys())
    body: dict = {
        "backend": dispatcher.backend,
        "registered_tasks": registered,
        "task_count": len(registered),
        "celery_active": None,
    }
    if dispatcher.backend == "celery":
        try:
            from app.worker.tasks import _celery

            inspector = _celery.control.inspect()
            active = inspector.active() or {}
            body["celery_active"] = {
                worker: len(tasks) for worker, tasks in active.items()
            }
        except Exception as exc:  # noqa: BLE001 - broker inspection is best-effort
            body["celery_active_error"] = str(exc)
    return body


@router.get("/api/v1/applications/{application_id}/report", tags=["applications"])
def download_report(application_id: str, format: str = "json") -> FileResponse:
    """Serve the generated allocation report file (json or pdf)."""
    if format not in ("json", "pdf"):
        raise HTTPException(status_code=400, detail="format must be json or pdf")
    settings = get_settings()
    result = load_result(application_id, settings)
    if result is None:
        record = get_record(application_id)
        if record is None or record.result is None:
            raise HTTPException(status_code=404, detail="no report available")
        result = record.result
    path = result.report_pdf_path if format == "pdf" else result.report_json_path
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        raise HTTPException(status_code=404, detail="report file missing on disk")
    media = "application/pdf" if format == "pdf" else "application/json"
    return FileResponse(file, media_type=media, filename=file.name)


@router.get("/api/v1/applications/{application_id}/documents", tags=["applications"])
def document_status(application_id: str) -> dict:
    """Return the current document-control snapshot for an application."""
    from app.agents.document_agent import DocumentControlAgent
    from app.engine.store import load_application

    settings = get_settings()
    application = load_application(application_id, settings)
    if application is None:
        raise HTTPException(status_code=404, detail="application not found")
    check = DocumentControlAgent().check(application)
    return {
        "application_id": application_id,
        "present": check.present,
        "missing": [req.code for req in check.missing],
        "complete": check.complete,
        "request_draft": check.request_draft,
    }


@router.get("/api/v1/applications/{application_id}/scorecard", tags=["applications"])
def application_scorecard(application_id: str) -> dict:
    """Return the deterministic BDDK-style composite risk scorecard."""
    from app.engine.scorecard import build_scorecard

    settings = get_settings()
    result = load_result(application_id, settings)
    if result is None:
        record = get_record(application_id)
        if record is None or record.result is None:
            raise HTTPException(status_code=404, detail="no decision available")
        result = record.result
    if result.decision is None:
        raise HTTPException(status_code=404, detail="no decision available")
    return build_scorecard(application_id, result.decision.factors).model_dump(mode="json")


@router.get("/api/v1/applications/{application_id}/offer", tags=["applications"])
def application_offer(application_id: str) -> dict:
    """Return the priced loan offer for an approved application."""
    from app.engine.offer import build_offer
    from app.engine.scorecard import build_scorecard

    settings = get_settings()
    result = load_result(application_id, settings)
    if result is None:
        record = get_record(application_id)
        if record is None or record.result is None:
            raise HTTPException(status_code=404, detail="no decision available")
        result = record.result
    if result.decision is None:
        raise HTTPException(status_code=404, detail="no decision available")
    if result.status != ApplicationStatus.APPROVED:
        raise HTTPException(status_code=409, detail="offer available only for approved applications")
    scorecard = build_scorecard(application_id, result.decision.factors)
    return build_offer(application_id, result.decision, scorecard).model_dump(mode="json")


@router.get("/api/v1/applications/{application_id}/documents/{code}/chunks", tags=["applications"])
def document_chunks(
    application_id: str,
    code: str,
    query: str = "",
    k: int = 3,
) -> dict:
    """Retrieve top-k RAG chunks from an uploaded applicant document."""
    from pathlib import Path

    from app.agents.document_agent import RAGDocumentAnalyzer

    settings = get_settings()
    normalized = code.upper()
    matched: str | None = None
    uploads = settings.uploads_dir
    if uploads.is_dir():
        for file in sorted(uploads.iterdir()):
            if file.is_file() and Path(file.name).stem.upper() == normalized:
                matched = str(file)
                break
    if matched is None:
        return {
            "application_id": application_id,
            "code": normalized,
            "matched_file": None,
            "chunks": [],
        }
    analyzer = RAGDocumentAnalyzer()
    analysis = analyzer.analyze([matched])
    hits = analyzer.retrieve(query, k=min(max(k, 1), 20)) if query else []
    return {
        "application_id": application_id,
        "code": normalized,
        "matched_file": matched,
        "total_chunks": analysis.total_chunks,
        "chunks": [chunk.model_dump(mode="json") for chunk in hits],
    }


@router.get("/api/v1/notifications", tags=["system"])
def list_notifications_api(limit: int = 50) -> dict:
    """Return the notification outbox (undelivered first)."""
    from app.engine.notifier import list_notifications

    settings = get_settings()
    entries = list_notifications(limit=limit, settings=settings)
    return {
        "total": len(entries),
        "pending": sum(1 for e in entries if not e.delivered),
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }


@router.post("/api/v1/notifications/{notification_id}/deliver", tags=["system"])
def deliver_notification(notification_id: str) -> dict:
    """Mark one outbox notification as delivered."""
    from app.engine.notifier import mark_delivered

    settings = get_settings()
    ok = mark_delivered(notification_id, settings)
    if not ok:
        raise HTTPException(status_code=404, detail="notification not found or already delivered")
    return {"notification_id": notification_id, "delivered": True}


@router.post("/api/v1/notifications/dispatch", tags=["system"])
def dispatch_notifications_api() -> dict:
    """Enqueue draining of pending outbox notifications."""
    receipt = TaskDispatcher().enqueue("app.tasks.dispatch_notifications")
    return receipt.model_dump(mode="json")
