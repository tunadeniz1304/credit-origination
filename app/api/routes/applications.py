"""Application lifecycle endpoints (applicant portal + staff views)."""

from __future__ import annotations

from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, db_session, load_application, require_staff
from app.api.schemas import ApplicationCreate, ObjectionRequest, OpenBankingConsentRequest
from app.api.views import (
    applicant_view,
    application_summary,
    cashflow_view,
    decision_view,
    document_view,
    offer_view,
    timeline_view,
)
from app.core.config import get_settings
from app.core.limiter import limiter
from app.core.rules import load_workflow
from app.core.security import Principal
from app.db.audit import entity_trail
from app.db.models import Application, CashflowFeatures, Document, Objection, utcnow
from app.documents.storage import UploadRejected
from app.workflow.dispatch import dispatch_processing
from app.workflow.offers import accept_offer, cancel_application
from app.workflow.pipeline import latest_decision, latest_offer
from app.workflow.service import ApplicationService, required_documents
from app.workflow.states import InvalidTransitionError, State

router = APIRouter(prefix="/api/v1/applications", tags=["applications"])


def _detail(session: Session, app: Application, user: Principal) -> dict:
    decision = latest_decision(session, app.id)
    offer = latest_offer(session, app.id)
    service = ApplicationService(session, user)
    return {
        **application_summary(app),
        "applicant": applicant_view(app),
        "declared_income": app.declared_income,
        "employment_type": app.employment_type,
        "employer_name": app.employer_name,
        "required_documents": [
            {"code": c, "description": d} for c, d in required_documents(app.employment_type)
        ],
        "missing_documents": [c for c, _ in service.missing_documents(app)],
        "consents": service.active_consents(app),
        "timeline": timeline_view(app),
        "decision": decision_view(decision, user) if decision else None,
        "offer": offer_view(offer) if offer else None,
        "letters": app.letters or {},
        "kyc": app.kyc if user.is_staff else None,
        "retry_count": app.retry_count,
        "last_error": app.last_error if user.is_staff else None,
        "task_id": app.task_id,
        "queue_backend": app.queue_backend,
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(get_settings().rate_limit_submit)
def create_application(
    request: Request,
    body: ApplicationCreate,
    background: BackgroundTasks,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    if user.role not in ("basvuran", "uzman", "kidemli_uzman", "komite", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bu işlem için yetkiniz yok")
    service = ApplicationService(session, user)
    app = service.create(body.model_dump())
    session.commit()
    dispatch_processing(session, app, background)
    return _detail(session, app, user)


@router.get("")
def list_applications(
    state: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    query = select(Application).order_by(Application.created_at.desc(), Application.id.desc())
    if user.role == "basvuran":
        query = query.where(Application.owner_user_id == user.user_id)
    elif not user.is_staff:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bu işlem için yetkiniz yok")
    if state:
        query = query.where(Application.state == state)
    rows = session.execute(query.limit(limit)).scalars().all()
    return {"applications": [application_summary(a) for a in rows], "count": len(rows)}


@router.get("/{application_id}")
def get_application(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    return _detail(session, load_application(session, application_id, user), user)


@router.get("/{application_id}/timeline")
def timeline(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    return {"application_id": app.id, "state": app.state, "events": timeline_view(app)}


@router.post("/{application_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    application_id: str,
    background: BackgroundTasks,
    code: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    if app.state not in (
        State.TASLAK.value,
        State.GONDERILDI.value,
        State.BELGE_BEKLENIYOR.value,
        State.UZMAN_INCELEMESI.value,
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "bu aşamada belge yüklenemez")
    settings = get_settings()
    data = await file.read(settings.max_upload_bytes + 1)
    service = ApplicationService(session, user)
    try:
        document = service.add_document(app, code, file.filename or "belge", data)
    except UploadRejected as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    missing = [c for c, _ in service.missing_documents(app)]
    session.commit()
    resumed = False
    if app.state == State.BELGE_BEKLENIYOR.value and not missing:
        dispatch_processing(session, app, background)
        resumed = True
    return {
        "application_id": app.id,
        "document": document_view(session, document, user),
        "missing_documents": missing,
        "complete": not missing,
        "processing_resumed": resumed,
    }


@router.get("/{application_id}/documents")
def list_documents(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    service = ApplicationService(session, user)
    return {
        "application_id": app.id,
        "documents": [document_view(session, d, user) for d in service.documents(app)],
        "missing_documents": [c for c, _ in service.missing_documents(app)],
    }


def _document(session: Session, app: Application, document_id: str) -> Document:
    document = session.get(Document, document_id)
    if document is None or document.application_id != app.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "belge bulunamadı")
    return document


@router.get("/{application_id}/documents/{document_id}/file")
def document_file(
    application_id: str,
    document_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> FileResponse:
    app = load_application(session, application_id, user)
    document = _document(session, app, document_id)
    path = Path(document.stored_path)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "dosya bulunamadı")
    return FileResponse(path, media_type=document.mime, filename=document.filename)


@router.get("/{application_id}/documents/{document_id}/chunks")
def document_chunks(
    application_id: str,
    document_id: str,
    query: str = "",
    k: int = Query(default=3, ge=1, le=20),
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict:
    """RAG retrieval restricted to one document of one application (bug #1)."""
    from app.agents.rag import ApplicationDocumentIndex

    app = load_application(session, application_id, user)
    document = _document(session, app, document_id)
    index = ApplicationDocumentIndex.for_documents(app.id, [document])
    hits = index.search(query, k=k) if query else []
    return {
        "application_id": app.id,
        "document_id": document.id,
        "total_chunks": index.size,
        "chunks": hits,
    }


@router.post("/{application_id}/consents/open-banking", status_code=status.HTTP_201_CREATED)
def open_banking_consent(
    application_id: str,
    body: OpenBankingConsentRequest,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    consent = ApplicationService(session, user).grant_consent(
        app, "ACIK_BANKACILIK", scope=body.scopes, days=body.days
    )
    return {
        "application_id": app.id,
        "consent": {
            "type": consent.type,
            "scope": consent.scope,
            "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
        },
    }


@router.get("/{application_id}/cashflow")
def cashflow(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    row = (
        session.execute(
            select(CashflowFeatures)
            .where(CashflowFeatures.application_id == app.id)
            .order_by(CashflowFeatures.id.desc())
        )
        .scalars()
        .first()
    )
    return {"application_id": app.id, **cashflow_view(row)}


@router.get("/{application_id}/decision")
def decision(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    found = latest_decision(session, app.id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "henüz karar yok")
    return decision_view(found, user)


@router.get("/{application_id}/offer")
def offer(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    found = latest_offer(session, app.id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "teklif yok")
    return offer_view(found)


@router.get("/{application_id}/schedule")
def schedule(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    found = latest_offer(session, app.id)
    if found is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "ödeme planı yalnızca onaylı/teklifli başvurularda vardır"
        )
    return {"application_id": app.id, **offer_view(found)}


@router.post("/{application_id}/offer/accept")
def accept(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    if user.role != "basvuran" and not user.is_staff:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bu işlem için yetkiniz yok")
    try:
        offer_row = accept_offer(session, app, user)
    except (InvalidTransitionError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {
        "application_id": app.id,
        "state": app.state,
        "offer": offer_view(offer_row, include_schedule=False),
        "contract": app.reports.get("contract"),
    }


@router.post("/{application_id}/cancel")
def cancel(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    app = load_application(session, application_id, user)
    try:
        cancel_application(session, app, user)
    except InvalidTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"application_id": app.id, "state": app.state}


@router.post("/{application_id}/objection", status_code=status.HTTP_201_CREATED)
def objection(
    application_id: str,
    body: ObjectionRequest,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> dict:
    """KVKK art. 11: object to an automated adverse decision → human review."""
    from datetime import timedelta

    app = load_application(session, application_id, user)
    if app.state not in (State.OTOMATIK_RET.value, State.REDDEDILDI.value):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "itiraz yalnızca olumsuz sonuçlanan başvurulara yapılabilir"
        )
    workflow = load_workflow()
    if app.decided_at is not None:
        decided = (
            app.decided_at
            if app.decided_at.tzinfo
            else app.decided_at.replace(tzinfo=utcnow().tzinfo)
        )
        if utcnow() - decided > timedelta(days=workflow.objection_window_days):
            raise HTTPException(status.HTTP_409_CONFLICT, "itiraz süresi dolmuştur")
    service = ApplicationService(session, user)
    record = Objection(
        application_id=app.id,
        reason=body.reason,
        due_at=utcnow() + timedelta(hours=workflow.sla_hours["ITIRAZ_INCELEMESI"]),
    )
    session.add(record)
    service.transition(app, State.ITIRAZ_INCELEMESI, "KVKK m.11 itirazı")
    session.flush()
    return {
        "application_id": app.id,
        "objection_id": record.id,
        "state": app.state,
        "due_at": record.due_at.isoformat(),
    }


@router.get("/{application_id}/report")
def report(
    application_id: str,
    format: str = Query(default="pdf", pattern="^(pdf|json)$"),
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> FileResponse:
    app = load_application(session, application_id, user)
    path = (app.reports or {}).get(format)
    if not path or not Path(path).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "rapor henüz oluşturulmadı")
    media = "application/pdf" if format == "pdf" else "application/json"
    return FileResponse(path, media_type=media, filename=Path(path).name)


@router.get("/{application_id}/contract")
def contract(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(current_user),
) -> FileResponse:
    app = load_application(session, application_id, user)
    path = (app.reports or {}).get("contract")
    if not path or not Path(path).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sözleşme hazır değil")
    return FileResponse(path, media_type="application/pdf", filename=Path(path).name)


@router.get("/{application_id}/audit")
def audit_trail(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict:
    app = load_application(session, application_id, user)
    return {
        "application_id": app.id,
        "entries": [
            {
                "id": e.id,
                "ts": e.ts.isoformat(),
                "actor": e.actor,
                "action": e.action,
                "payload": e.payload,
                "hash": e.hash,
                "prev_hash": e.prev_hash,
            }
            for e in entity_trail(session, app.id)
        ],
    }


@router.get("/{application_id}/network")
def network(
    application_id: str,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict:
    app = load_application(session, application_id, user)
    ring = (app.kyc or {}).get("ring") or {
        "nodes": [],
        "edges": [],
        "ring_size": 1,
        "flagged": False,
    }
    return {"application_id": app.id, **ring}


@router.post("/{application_id}/reprocess")
def reprocess(
    application_id: str,
    background: BackgroundTasks,
    session: Session = Depends(db_session),
    user: Principal = Depends(require_staff),
) -> dict:
    app = load_application(session, application_id, user)
    if app.state not in (
        State.GONDERILDI.value,
        State.BELGE_BEKLENIYOR.value,
        State.BELGE_INCELEMEDE.value,
        State.VERI_TOPLANIYOR.value,
        State.KARAR_MOTORU.value,
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "bu aşamadaki başvuru yeniden işlenemez")
    ApplicationService(session, user).audit("APPLICATION_REPROCESSED", app.id, {"state": app.state})
    session.commit()
    dispatch_processing(session, app, background)
    return {"application_id": app.id, "state": app.state, "queue_backend": app.queue_backend}
