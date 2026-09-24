"""Kredi Tahsis Raporu / Memorandumu — JSON (machine) + PDF (human).

The PDF embeds DejaVu Sans so Turkish characters (ı, ş, ğ, İ, ç, ö, ü) render
and extract correctly. The report records the decision, its reason codes,
rule results, limits, pricing and schedule, the committee summary and the
applicant letter, plus the reproducibility stamp (rule set, model and feature
snapshot hash).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import Settings
from app.core.crypto import mask_tckn
from app.core.logging import get_logger
from app.core.pdf import FONT_BOLD, FONT_REGULAR, register_fonts
from app.db.models import Application, Decision, Offer
from app.decisioning.features import FEATURE_LABELS
from app.workflow.states import STATE_LABELS, State

_NAVY = colors.HexColor("#1F3B57")
OUTCOME_COLORS = {
    "OTOMATIK_ONAY": colors.HexColor("#1E7B34"),
    "OTOMATIK_RET": colors.HexColor("#B3261E"),
    "UZMAN_INCELEMESI": colors.HexColor("#9A6700"),
}


def _tl(value: float | None) -> str:
    if value is None:
        return "—"
    text = f"{value:,.2f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") + " TL"


def _pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    text = f"{value * 100:,.{decimals}f}"
    return "%" + text.replace(",", "_").replace(".", ",").replace("_", ".")


def _styles() -> dict[str, ParagraphStyle]:
    register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=17,
            textColor=_NAVY,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            textColor=_NAVY,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "b",
            parent=base["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "s",
            parent=base["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#52606D"),
        ),
        "cell": ParagraphStyle(
            "c", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=8.5, leading=11
        ),
    }


def _table(rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D3DE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),
        ]
    table.setStyle(TableStyle(style))
    return table


def _para_block(text: str, style: ParagraphStyle) -> list[Any]:
    return [Paragraph(escape(line) or "&nbsp;", style) for line in text.splitlines()]


def report_payload(
    app: Application, decision: Decision, offer: Offer | None, pii: dict[str, Any]
) -> dict[str, Any]:
    return {
        "application_id": app.id,
        "state": app.state,
        "applicant": {
            "name": pii.get("name"),
            "identity_no_masked": mask_tckn(pii.get("identity_no")),
        },
        "request": {
            "product": app.product,
            "amount": app.requested_amount,
            "term_months": app.requested_term_months,
            "declared_income": app.declared_income,
        },
        "decision": {
            "id": decision.id,
            "kind": decision.kind,
            "outcome": decision.outcome,
            "conditional": decision.conditional,
            "pd": decision.pd,
            "risk_band": decision.risk_band,
            "score_points": decision.score_points,
            "reason_codes": decision.reason_codes,
            "rule_results": decision.rule_results,
            "limits": decision.limits,
            "counterfactuals": decision.counterfactuals,
            "challenger": decision.challenger,
            "versions": {"rule_set": decision.rule_set_version, "model": decision.model_version},
            "feature_hash": decision.feature_hash,
            "decided_by": decision.decided_by,
        },
        "pricing": {k: v for k, v in (decision.pricing or {}).items() if k != "schedule"},
        "offer": None
        if offer is None
        else {
            "amount": offer.amount,
            "term_months": offer.term_months,
            "annual_rate": offer.annual_rate,
            "instalment": offer.instalment,
            "apr": offer.apr,
            "valid_until": offer.valid_until.isoformat(),
        },
        "narratives": decision.narratives,
    }


def generate_report(
    app: Application,
    decision: Decision,
    offer: Offer | None,
    pii: dict[str, Any],
    settings: Settings,
) -> dict[str, str]:
    directory = settings.report_dir / app.id
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{app.id}_rapor.json"
    json_path.write_text(
        json.dumps(
            report_payload(app, decision, offer, pii), ensure_ascii=False, indent=2, default=str
        ),
        encoding="utf-8",
    )
    pdf_path = directory / f"{app.id}_rapor.pdf"
    build_pdf(app, decision, offer, pii, pdf_path)
    get_logger("reports.credit").info("credit report written for %s", app.id)
    return {"json": str(json_path), "pdf": str(pdf_path)}


def build_pdf(
    app: Application, decision: Decision, offer: Offer | None, pii: dict[str, Any], path: Path
) -> None:
    st = _styles()
    narratives = decision.narratives or {}
    pricing = decision.pricing or {}
    limits = decision.limits or {}
    outcome_label = (
        STATE_LABELS.get(State(decision.outcome), decision.outcome)
        if decision.outcome in State.__members__
        else decision.outcome
    )
    story: list[Any] = [
        Paragraph("KREDİ TAHSİS MEMORANDUMU", st["title"]),
        Paragraph(
            escape(f"Başvuru No: {app.id} · Ürün: {app.product} · Karar türü: {decision.kind}"),
            st["small"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("1. Başvuru Bilgileri", st["h2"]),
        _table(
            [
                ["Başvuru Sahibi", pii.get("name") or "—"],
                ["T.C. Kimlik No", mask_tckn(pii.get("identity_no"))],
                ["Beyan Edilen Aylık Net Gelir", _tl(app.declared_income)],
                [
                    "Talep Edilen Tutar / Vade",
                    f"{_tl(app.requested_amount)} / {app.requested_term_months} ay",
                ],
                ["Çalışma Şekli / İşveren", f"{app.employment_type} / {app.employer_name or '—'}"],
            ],
            [6 * cm, 11 * cm],
            header=False,
        ),
        Paragraph("2. Karar Özeti", st["h2"]),
    ]
    verdict = ParagraphStyle(
        "v",
        parent=st["body"],
        fontName=FONT_BOLD,
        fontSize=13,
        textColor=OUTCOME_COLORS.get(decision.outcome, _NAVY),
    )
    story.append(
        Paragraph(
            escape(
                f"KARAR: {outcome_label}"
                + (" (koşullu / karşı teklif)" if decision.conditional else "")
            ),
            verdict,
        )
    )
    story.append(
        _table(
            [
                ["Gösterge", "Değer"],
                ["12 aylık temerrüt olasılığı (PD)", _pct(decision.pd, 2)],
                ["Risk bandı", decision.risk_band or "—"],
                ["Skor kartı puanı (300–900)", f"{decision.score_points or 0:.0f}"],
                [
                    "Borç servis oranı (talep / teklif)",
                    f"{_pct(limits.get('dsr_requested'))} / {_pct(limits.get('dsr_offer'))}",
                ],
                ["DSR politika sınırı", _pct(limits.get("max_dsr"))],
                ["Onaylanabilir azami tutar", _tl(limits.get("max_approvable_amount"))],
                ["Challenger PD (gölge)", _pct((decision.challenger or {}).get("pd"), 2)],
            ],
            [9 * cm, 8 * cm],
        )
    )
    if decision.reason_codes:
        story.append(Paragraph("3. Gerekçe Kodları", st["h2"]))
        rows = [["Kod", "Açıklama", "Kaynak"]]
        rows += [
            [r["code"], Paragraph(escape(r["text"]), st["cell"]), r.get("source", "")]
            for r in decision.reason_codes
        ]
        story.append(_table(rows, [4.2 * cm, 11 * cm, 1.8 * cm]))
    fired = [r for r in decision.rule_results if r.get("fired")]
    story.append(Paragraph("4. Politika Kuralları", st["h2"]))
    if fired:
        rows = [["Kural", "Açıklama", "Aksiyon"]]
        rows += [
            [r["id"], Paragraph(escape(r["description"]), st["cell"]), r["action"]] for r in fired
        ]
        story.append(_table(rows, [4 * cm, 10.5 * cm, 2.5 * cm]))
    else:
        story.append(Paragraph("Tetiklenen politika kuralı bulunmamaktadır.", st["body"]))
    shap = (decision.explanation or {}).get("shap", {})
    if shap:
        story.append(Paragraph("5. Model Açıklaması (SHAP, log-odds katkısı)", st["h2"]))
        rows = [["Değişken", "Katkı", "Etki"]]
        for feature, value in list(shap.items())[:8]:
            rows.append(
                [
                    FEATURE_LABELS.get(feature, feature),
                    f"{value:+.3f}",
                    "Riski artırıyor" if value > 0 else "Riski azaltıyor",
                ]
            )
        story.append(_table(rows, [8 * cm, 3 * cm, 6 * cm]))
    if pricing:
        story.append(Paragraph("6. Fiyatlama ve Teklif", st["h2"]))
        story.append(
            _table(
                [
                    ["Kalem", "Değer"],
                    [
                        "Kredi tutarı / vade",
                        f"{_tl(pricing.get('amount'))} / {pricing.get('term_months')} ay",
                    ],
                    ["Yıllık akdi faiz", _pct(pricing.get("annual_rate"), 2)],
                    ["Aylık taksit (BSMV+KKDF dahil)", _tl(pricing.get("instalment"))],
                    ["Yıllık maliyet oranı (efektif)", _pct(pricing.get("apr"), 2)],
                    ["Toplam geri ödeme", _tl(pricing.get("total_payment"))],
                    ["Beklenen kayıp (PD×LGD×EAD)", _tl(pricing.get("expected_loss_annual"))],
                    [
                        "RAROC / hedef",
                        f"{_pct(pricing.get('raroc'))} / {_pct(pricing.get('target_raroc'))}",
                    ],
                ],
                [9 * cm, 8 * cm],
            )
        )
        schedule = pricing.get("schedule") or []
        if schedule:
            rows = [["Ay", "Taksit", "Faiz", "KKDF", "BSMV", "Anapara", "Kalan"]]
            for row in schedule[:12]:
                rows.append(
                    [
                        str(row["period"]),
                        _tl(row["instalment"]),
                        _tl(row["interest"]),
                        _tl(row["kkdf"]),
                        _tl(row["bsmv"]),
                        _tl(row["principal"]),
                        _tl(row["closing_balance"]),
                    ]
                )
            story.append(Spacer(1, 0.2 * cm))
            story.append(
                _table(rows, [1.1 * cm, 2.6 * cm, 2.5 * cm, 2 * cm, 2 * cm, 2.6 * cm, 2.8 * cm])
            )
    if decision.counterfactuals:
        story.append(Paragraph("7. Karşı-olgusal Öneriler", st["h2"]))
        for cf in decision.counterfactuals:
            story.append(Paragraph(escape("• " + cf["text"]), st["body"]))
    if narratives.get("committee_summary"):
        story.append(Paragraph("8. Komite Özeti", st["h2"]))
        story += _para_block(narratives["committee_summary"], st["body"])
    if narratives.get("applicant_letter"):
        story.append(Paragraph("9. Başvurana Bildirim Mektubu", st["h2"]))
        story += _para_block(narratives["applicant_letter"], st["body"])
    story += [
        Spacer(1, 0.4 * cm),
        Paragraph(
            escape(
                f"Yeniden üretilebilirlik: kural seti {decision.rule_set_version} · model {decision.model_version} · "
                f"özellik özeti {decision.feature_hash[:16]}… · anlatı modu {narratives.get('mode', 'demo')}. "
                "Bağlayıcı karar deterministik karar motorundan gelir; LLM yalnızca açıklama üretir."
            ),
            st["small"],
        ),
    ]
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"Kredi Tahsis Memorandumu {app.id}",
        author="Anil2 Kredi Tahsis Platformu",
    )
    doc.build(story)


def pdf_text(path: str | Path) -> str:
    import fitz

    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)
