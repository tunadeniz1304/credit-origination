"""Offer acceptance, pre-contract information form, disbursement, cancellation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.pdf import FONT_BOLD, FONT_REGULAR, register_fonts
from app.core.rules import load_policy_file
from app.core.security import Principal
from app.db.models import Application, LoanPerformance, Offer, utcnow
from app.workflow.pipeline import latest_offer
from app.workflow.service import ApplicationService
from app.workflow.states import State


def _tl(value: float) -> str:
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") + " TL"


def build_contract_form(app: Application, offer: Offer, pii: dict[str, Any], path: Path) -> None:
    """Sözleşme Öncesi Bilgi Formu (summary; synthetic, not legal advice)."""
    register_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setTitle(f"Sözleşme Öncesi Bilgi Formu {app.id}")
    _, height = A4
    pdf.setFont(FONT_BOLD, 14)
    pdf.drawString(50, height - 60, "TÜKETİCİ KREDİSİ SÖZLEŞME ÖNCESİ BİLGİ FORMU")
    pdf.setFont(FONT_REGULAR, 10)
    rows = [
        ("Başvuru No", app.id),
        ("Kredi Kullanan", pii.get("name") or ""),
        ("Kredi Tutarı", _tl(offer.amount)),
        ("Vade", f"{offer.term_months} ay"),
        ("Yıllık Akdi Faiz Oranı", f"%{offer.annual_rate * 100:.2f}".replace(".", ",")),
        ("Aylık Taksit (BSMV ve KKDF dahil)", _tl(offer.instalment)),
        ("Tahsis Ücreti (BSMV dahil)", _tl(offer.fees)),
        ("Toplam Geri Ödeme", _tl(offer.total_payment)),
        ("Yıllık Maliyet Oranı (efektif)", f"%{offer.apr * 100:.2f}".replace(".", ",")),
        ("Cayma Hakkı", "Kredi kullandırımından itibaren 14 gün"),
    ]
    y = height - 100
    for label, value in rows:
        pdf.drawString(60, y, f"{label}:")
        pdf.drawString(300, y, value)
        y -= 20
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawString(
        60,
        y - 20,
        "Bu form sentetik demo amaçlıdır; 6502 sayılı Kanun kapsamındaki resmi form yerine geçmez.",
    )
    pdf.save()


def accept_offer(session: Session, app: Application, user: Principal) -> Offer:
    offer = latest_offer(session, app.id)
    if offer is None or offer.status != "SUNULDU":
        raise ValueError("kabul edilebilir teklif yok")
    valid_until = (
        offer.valid_until
        if offer.valid_until.tzinfo
        else offer.valid_until.replace(tzinfo=utcnow().tzinfo)
    )
    if valid_until < utcnow():
        offer.status = "SURESI_DOLDU"
        raise ValueError("teklifin geçerlilik süresi dolmuş")
    service = ApplicationService(session, user)
    service.transition(app, State.TEKLIF_KABUL, "teklif kabul edildi")
    offer.status = "KABUL"
    offer.accepted_at = utcnow()
    settings = get_settings()
    path = settings.report_dir / app.id / f"{app.id}_sozlesme_oncesi_bilgi_formu.pdf"
    build_contract_form(app, offer, service.pii(app), path)
    app.reports = {**(app.reports or {}), "contract": str(path)}
    service.transition(app, State.SOZLESME_HAZIR, "sözleşme öncesi bilgi formu hazırlandı")
    return offer


def disburse(session: Session, app: Application, user: Principal) -> None:
    service = ApplicationService(session, user)
    service.transition(app, State.KULLANDIRILDI, "kredi kullandırıldı")
    offer = latest_offer(session, app.id)
    if offer is not None:
        seed_performance(session, app, offer)


def seed_performance(session: Session, app: Application, offer: Offer, months: int = 6) -> None:
    """Synthetic post-disbursement behaviour for the early-warning system."""
    from app.ews.model import simulate_behaviour
    from app.workflow.pipeline import latest_decision

    decision = latest_decision(session, app.id)
    pd = (
        decision.pd if decision and decision.pd else load_policy_file().decision.auto_approve_max_pd
    )
    for row in simulate_behaviour(app.id, offer.amount, offer.instalment, pd, months):
        session.add(LoanPerformance(application_id=app.id, **row))


def cancel_application(session: Session, app: Application, user: Principal) -> None:
    ApplicationService(session, user).transition(app, State.IPTAL, "başvuru iptal edildi")
