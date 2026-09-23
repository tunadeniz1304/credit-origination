"""Kredi Tahsis Raporu generation: JSON (machine) + PDF (human)."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import Settings
from app.core.logging import get_logger
from app.models import PipelineResult

_RED = colors.HexColor("#B3261E")
_GREEN = colors.HexColor("#1E7B34")


def generate_report_files(
    payload: PipelineResult,
    application_id: str,
    settings: Settings,
) -> tuple[str, str]:
    """Write ``{app_id}_report.json`` and ``{app_id}_report.pdf``; return paths."""
    logger = get_logger("engine.reports")
    report_dir = settings.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"{application_id}_report.json"
    json_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    pdf_path = report_dir / f"{application_id}_report.pdf"
    _build_pdf(payload, application_id, pdf_path)

    logger.info(
        "Kredi Tahsis Raporu written for %s -> %s / %s", application_id, json_path, pdf_path
    )
    return str(json_path), str(pdf_path)


def _build_pdf(payload: PipelineResult, application_id: str, path) -> None:
    """Compose the BDDK-style allocation report as an A4 PDF."""
    logger = get_logger("engine.reports")
    application = payload.application
    applicant = application.applicant
    decision = payload.decision

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=18,
                              alignment=TA_CENTER, spaceAfter=22,
                              textColor=colors.HexColor("#1F3B57")))
    styles.add(ParagraphStyle(name="SectionHeading", parent=styles["Heading2"],
                              textColor=colors.HexColor("#1F3B57"),
                              spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyJustified", parent=styles["BodyText"],
                              alignment=TA_JUSTIFY, leading=14))
    styles.add(ParagraphStyle(
        name="Verdict", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=14,
        fontName="Helvetica-Bold",
        textColor=_GREEN if payload.status.value == "APPROVED" else _RED, spaceBefore=10,
    ))

    story = [
        Paragraph("KREDİ TAHSİS RAPORU", styles["ReportTitle"]),
        Paragraph(
            f"Başvuru No: {application_id}<br/>Belge: Kredi Tahsis Komitesi Karar Raporu",
            styles["BodyJustified"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("1. Başvuru Bilgileri", styles["SectionHeading"]),
        Table(
            [
                ["Başvuru Sahibi", applicant.name],
                ["T.C. Kimlik No", applicant.identity_no],
                ["Aylık Gelir", f"{applicant.monthly_income:,.2f} TRY"],
                ["Talep Edilen Kredi", f"{application.requested_amount:,.2f} {application.currency}"],
                ["Vade", f"{application.requested_term_months} ay"],
            ],
            colWidths=[6 * cm, 10 * cm],
        ),
        Spacer(1, 0.4 * cm),
    ]

    if decision is not None:
        factor_rows = [["Değerlendirme Kriteri", "Gözlenen", "Sınır", "Sonuç"]]
        for factor in decision.factors:
            factor_rows.append(
                [
                    factor.name,
                    f"{factor.value:,.1f}{factor.unit}",
                    f"{factor.operator} {factor.threshold:g}{factor.unit}",
                    "UYGUN" if factor.passed else "AYKIRI",
                ]
            )
        story += [
            PageBreak(),
            Paragraph("2. Komite Eşik Değerlendirmeleri", styles["SectionHeading"]),
            Table(factor_rows, colWidths=[6 * cm, 4 * cm, 4 * cm, 2.5 * cm]),
            Spacer(1, 0.6 * cm),
            Paragraph(
                f"Önerilen Kredi Tutarı: {decision.suggested_amount:,.2f} TRY<br/>"
                f"Önerilen Vade: {decision.suggested_term_months} ay",
                styles["BodyJustified"],
            ),
            Spacer(1, 0.4 * cm),
            Paragraph("3. Gerekçe ve Değerlendirme", styles["SectionHeading"]),
            Paragraph(decision.rationale.replace("\n", "<br/>"), styles["BodyJustified"]),
            Spacer(1, 0.4 * cm),
            Paragraph(f"KARAR: {payload.status.value}", styles["Verdict"]),
        ]
        _append_schedule(story, styles, application_id, decision)

    _table_style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D3DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for element in story:
        if isinstance(element, Table):
            element.setStyle(TableStyle(_table_style))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"Kredi Tahsis Raporu {application_id}",
        author="Akıllı Kredi Operasyon Ajanı",
    )
    try:
        doc.build(story)
    except Exception as exc:  # noqa: BLE001 - surface writer failures loudly
        logger.exception("PDF build failed for %s: %s", application_id, exc)
        raise


def _append_schedule(story, styles, application_id: str, decision) -> None:
    """Append the amortization plan (first 12 rows) to the report story."""
    from app.engine.schedule import build_schedule

    schedule = build_schedule(
        application_id,
        principal=decision.suggested_amount,
        term_months=decision.suggested_term_months,
    )
    rows = [["Ay", "Kalan Bakiye", "Faiz", "Ana Para", "Taksit"]]
    for row in schedule.rows[:12]:
        rows.append(
            [
                str(row.period),
                f"{row.principal_balance:,.2f}",
                f"{row.interest:,.2f}",
                f"{row.principal_paid:,.2f}",
                f"{row.instalment:,.2f}",
            ]
        )
    story += [
        PageBreak(),
        Paragraph("4. Geri Ödeme Planı (Vade Planı)", styles["SectionHeading"]),
        Table(rows, colWidths=[2 * cm, 4 * cm, 3.4 * cm, 3.4 * cm, 3.2 * cm]),
        Spacer(1, 0.4 * cm),
        Paragraph(
            f"Aylık Taksit: {schedule.instalment:,.2f} TRY &nbsp;&nbsp; "
            f"Toplam Ödeme: {schedule.total_payment:,.2f} TRY &nbsp;&nbsp; "
            f"Toplam Faiz: {schedule.total_interest:,.2f} TRY",
            styles["BodyJustified"],
        ),
    ]


def pdf_to_text(path: str) -> str:
    """Extract text from a generated PDF (used by tests / audits)."""
    try:
        import pypdf
    except ImportError:  # pragma: no cover - optional introspection path
        return ""
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

