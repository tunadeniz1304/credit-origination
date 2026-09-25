"""Producer-independent tamper detection (audit F05) and the OCR path (F06).

Test documents come from different tools: reportlab (the sample generator),
PyMuPDF (authored from scratch), pikepdf (metadata rewritten), "Word export"
metadata, a PyMuPDF white-out forgery whose producer is spoofed to look like
the genuine issuer, and a scanned page with a patched text layer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.documents.extraction import extract_fields, extract_text, ocr_status
from app.documents.fraud import _pdf_date, analyse_document, revision_count
from app.documents.samples import make_payslip

PAYSLIP_LINES = [
    ("ÜCRET BORDROSU", 16),
    ("Ad Soyad: Ayşe Kaya", 10),
    ("T.C. Kimlik No: 10000000146", 10),
    ("İşveren: Anadolu Bilişim Ltd. Şti.", 10),
    ("Dönem: 2026-08", 10),
    ("Brüt Ücret: 63.380,28 TL", 10),
    ("Toplam Kesinti: 18.380,28 TL", 10),
    ("Net Ücret: 45.000,00 TL", 10),
]


def _codes(path: Path, code: str = "INCOME") -> set[str]:
    text = extract_text(path, "application/pdf")
    fields = extract_fields(code, text, path)
    return {s.code for s in analyse_document(path, code, "application/pdf", fields, text).signals}


def _score(path: Path, code: str = "INCOME") -> float:
    text = extract_text(path, "application/pdf")
    fields = extract_fields(code, text, path)
    return analyse_document(path, code, "application/pdf", fields, text).score


@pytest.fixture()
def reportlab_payslip(tmp_path) -> Path:
    path = tmp_path / "bordro_reportlab.pdf"
    make_payslip(
        path,
        name="Ayşe Kaya",
        tckn="10000000146",
        employer="Anadolu Bilişim Ltd. Şti.",
        period="2026-08",
        gross=63_380.28,
    )
    return path


@pytest.fixture()
def pymupdf_payslip(tmp_path) -> Path:
    """A genuine payslip authored with a different library (PyMuPDF), one font."""
    import fitz

    from app.core.pdf import FONT_DIR

    path = tmp_path / "bordro_pymupdf.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for text, size in PAYSLIP_LINES:
        page.insert_text(
            (60, y),
            text,
            fontsize=size,
            fontname="dejavu",
            fontfile=str(FONT_DIR / "DejaVuSans.ttf"),
        )
        y += 24
    doc.set_metadata({"producer": "Logo Bordro Plus", "creator": "Logo Bordro Plus"})
    doc.save(str(path))
    doc.close()
    return path


def _set_info(path: Path, **info: str) -> None:
    import pikepdf

    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for key, value in info.items():
            pdf.docinfo[f"/{key}"] = value
        pdf.save(path)


def test_genuine_documents_from_different_tools_score_zero(reportlab_payslip, pymupdf_payslip):
    assert _score(reportlab_payslip) == 0.0
    assert _codes(pymupdf_payslip) == set()


def test_unknown_producer_alone_is_not_a_signal(reportlab_payslip):
    _set_info(reportlab_payslip, Producer="SomeVendor PDF Engine 9.1", Creator="HR Suite")
    assert _codes(reportlab_payslip) == set()


def test_word_export_metadata_on_system_document(reportlab_payslip):
    _set_info(
        reportlab_payslip,
        Producer="Microsoft® Word for Microsoft 365",
        Creator="Microsoft® Word for Microsoft 365",
    )
    assert _codes(reportlab_payslip) == {"producer_unexpected"}
    assert "producer_unexpected" in _codes(reportlab_payslip, "BANK_STATEMENT")


def test_pikepdf_metadata_rewrite_is_caught(reportlab_payslip):
    import pikepdf

    later = (datetime.now(UTC) + timedelta(days=5)).strftime("D:%Y%m%d%H%M%S")
    with pikepdf.open(reportlab_payslip, allow_overwriting_input=True) as pdf:
        created = pdf.docinfo["/CreationDate"]
        with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
            meta["pdf:Producer"] = "Original Payroll Engine"
            meta["xmp:CreatorTool"] = "Original Payroll Engine"
        pdf.docinfo["/CreationDate"] = created  # the forger keeps the issue date
        pdf.docinfo["/ModDate"] = later
        pdf.docinfo["/Producer"] = "Retouched"
        pdf.save(reportlab_payslip)
    codes = _codes(reportlab_payslip)
    assert {"modified_after_creation", "metadata_inconsistent"} <= codes
    assert "incremental_update" not in codes  # full rewrite, not an incremental save


def test_whiteout_forgery_is_caught_even_with_a_spoofed_producer(reportlab_payslip):
    """The forger copies the issuer's producer/creator strings; content signals remain."""
    import fitz

    doc = fitz.open(str(reportlab_payslip))
    page = doc[0]
    rects = page.search_for("45.311,19")  # the net salary printed by make_payslip
    assert rects
    for rect in rects:
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
        page.insert_text((rect.x0, rect.y1 - 2), "65.000,00", fontname="helv", fontsize=10)
    doc.set_metadata(
        {**(doc.metadata or {}), "producer": "ReportLab PDF Library - www.reportlab.com"}
    )
    doc.saveIncr()
    doc.close()
    codes = _codes(reportlab_payslip)
    assert {"incremental_update", "whiteout_overlay", "digit_font_mismatch"} <= codes
    # The covered value is still in the content stream, so arithmetic may look consistent;
    # detection must not depend on it.
    assert "editing_tool" not in codes and "producer_unexpected" not in codes
    assert _score(reportlab_payslip) >= 0.6


def test_scanned_page_with_patched_text_layer(reportlab_payslip, tmp_path):
    import fitz

    scan = tmp_path / "scan_patched.pdf"
    with fitz.open(str(reportlab_payslip)) as src:
        pix = src[0].get_pixmap(dpi=100)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((300, 500), "Net Ücret: 65.000,00 TL", fontname="helv", fontsize=10)
    doc.save(str(scan))
    doc.close()
    assert "text_over_image" in _codes(scan)


def test_pdf_date_parses_info_and_xmp_formats():
    assert _pdf_date("D:20260901120000+03'00'") == datetime(2026, 9, 1, 12, 0, 0)
    assert _pdf_date("2026-09-01T12:00:00+03:00") == datetime(2026, 9, 1, 12, 0, 0)
    assert _pdf_date("garbage") is None and _pdf_date(None) is None
    assert revision_count(b"%PDF-1.7 ... %%EOF ... %%EOF") == 2


# ------------------------------------------------------------------ OCR (F06)
def test_ocr_status_reports_why_ocr_is_unavailable(monkeypatch):
    import sys

    ocr_status.cache_clear()
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    try:
        assert ocr_status() == {"available": False, "reason": "pytesseract kurulu değil"}
    finally:
        ocr_status.cache_clear()


def test_ocr_status_with_binary_and_language(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        get_tesseract_version=lambda: "5.3.0", get_languages=lambda config="": ["eng"]
    )
    import sys

    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    ocr_status.cache_clear()
    try:
        assert ocr_status()["reason"] == "tesseract 'tur' dil paketi yok"
        fake.get_languages = lambda config="": ["eng", "tur"]
        ocr_status.cache_clear()
        assert ocr_status() == {"available": True, "reason": "", "version": "5.3.0"}
    finally:
        ocr_status.cache_clear()


def test_scanned_pdf_goes_to_ocr_or_manual_review(reportlab_payslip, tmp_path):
    from app.documents.samples import make_scanned_pdf

    scan = make_scanned_pdf(reportlab_payslip, tmp_path / "scan.pdf")
    result = extract_text(scan, "application/pdf")
    if not ocr_status()["available"]:
        assert result.source == "none" and not result.ocr_available
        pytest.skip(
            "Tesseract with the Turkish model is not installed: manual-review path verified"
        )
    assert result.source == "ocr"
    fields = {f.name: f for f in extract_fields("INCOME", result, scan)}
    assert "net_ucret" in fields and fields["net_ucret"].confidence < 0.95
