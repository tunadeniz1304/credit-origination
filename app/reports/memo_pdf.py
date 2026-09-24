"""PDF rendering of the AI credit memorandum (DejaVu Sans, Turkish glyphs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.core.pdf import FONT_BOLD, FONT_REGULAR, register_fonts


def build_memo_pdf(memo: Any, path: Path) -> None:
    register_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "t",
        parent=base["Title"],
        fontName=FONT_BOLD,
        fontSize=16,
        textColor=colors.HexColor("#1F3B57"),
    )
    h2 = ParagraphStyle(
        "h",
        parent=base["Heading2"],
        fontName=FONT_BOLD,
        fontSize=11.5,
        textColor=colors.HexColor("#1F3B57"),
    )
    body = ParagraphStyle(
        "b", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=9.5, leading=13
    )
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=colors.HexColor("#52606D"))
    story: list[Any] = [
        Paragraph("KREDİ TAHSİS MEMORANDUMU (AI ANALİST ÖNERİSİ)", title),
        Paragraph(
            escape(
                f"Başvuru: {memo.application_id} · Mod: {memo.mode} · Protokol: {memo.protocol}"
            ),
            small,
        ),
        Spacer(1, 0.3 * cm),
    ]
    for section in memo.sections:
        story.append(Paragraph(escape(section.title), h2))
        story.append(Paragraph(escape(section.text), body))
    story.append(Paragraph("Öneri", h2))
    story.append(Paragraph(escape(memo.recommendation), body))
    story.append(Spacer(1, 0.3 * cm))
    trail = " → ".join(step.tool for step in memo.steps)
    story.append(Paragraph(escape(f"Ajan adımları: {trail}"), small))
    story.append(
        Paragraph(
            "Bu memorandum öneri niteliğindedir; sayılar karar bağlamındaki alanlara atıflıdır. "
            "Bağlayıcı karar deterministik karar motoru ve yetkili kredi personeline aittir.",
            small,
        )
    )
    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    ).build(story)
