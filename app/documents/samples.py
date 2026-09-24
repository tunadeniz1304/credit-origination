"""Synthetic, realistic Turkish documents for demos and tests.

Generates payslips (bordro), bank statements (hesap özeti), e-Devlet barcoded
documents (yerleşim yeri / SGK hizmet dökümü) with a QR code, identity cards
and tax plates. :func:`tamper_pdf` produces deliberately manipulated copies:
an overlay in a different font saved as an *incremental update* with an
editing tool's producer string and a later modification date — exactly the
traces the fraud detector looks for.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.core.pdf import FONT_BOLD, FONT_REGULAR, register_fonts

_BARCODE_SALT = "anil2-edevlet-mock"
GENERATOR_CREATORS = {
    "INCOME": "e-Bordro Sistemi",
    "BANK_STATEMENT": "Anil Bank İnternet Şubesi",
    "ADDRESS": "e-Devlet Kapısı",
    "EMPLOYMENT": "e-Devlet Kapısı",
    "IDENTITY": "NVİ Kimlik Paylaşım",
    "TAX_PLATE": "GİB İnteraktif Vergi Dairesi",
}


def money(value: float) -> str:
    text = f"{value:,.2f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") + " TL"


def make_barcode(seed: str) -> str:
    body = str(int(hashlib.sha256(seed.encode()).hexdigest()[:10], 16) % 10**12).zfill(12)
    check = int(hashlib.sha256((_BARCODE_SALT + body).encode()).hexdigest(), 16) % 97
    return f"ED{body}{check:02d}"


def verify_barcode(barcode: str) -> bool:
    if len(barcode) != 16 or not barcode.startswith("ED") or not barcode[2:].isdigit():
        return False
    body, check = barcode[2:14], int(barcode[14:])
    return int(hashlib.sha256((_BARCODE_SALT + body).encode()).hexdigest(), 16) % 97 == check


def _canvas(path: Path, title: str, creator: str) -> canvas.Canvas:
    register_fonts()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setTitle(title)
    pdf.setAuthor(creator)
    pdf.setCreator(creator)
    pdf.setSubject("Sentetik demo belgesi — gerçek kişi verisi değildir")
    return pdf


def _header(pdf: canvas.Canvas, title: str, subtitle: str) -> float:
    width, height = A4
    pdf.setFont(FONT_BOLD, 15)
    pdf.drawString(50, height - 60, title)
    pdf.setFont(FONT_REGULAR, 9)
    pdf.drawString(50, height - 76, subtitle)
    pdf.line(50, height - 84, width - 50, height - 84)
    return height - 110


def _rows(pdf: canvas.Canvas, y: float, rows: list[tuple[str, str]]) -> float:
    for label, value in rows:
        pdf.setFont(FONT_REGULAR, 10)
        pdf.drawString(60, y, f"{label}:")
        pdf.drawString(250, y, value)
        y -= 18
    return y


def make_payslip(
    path: Path,
    *,
    name: str,
    tckn: str,
    employer: str,
    period: str,
    gross: float,
    net: float | None = None,
) -> dict[str, float]:
    """Payslip whose arithmetic is consistent unless ``net`` is forced."""
    sgk = round(gross * 0.14, 2)
    unemployment = round(gross * 0.01, 2)
    income_tax = round((gross - sgk - unemployment) * 0.15, 2)
    stamp = round(gross * 0.00759, 2)
    deductions = round(sgk + unemployment + income_tax + stamp, 2)
    true_net = round(gross - deductions, 2)
    net_value = true_net if net is None else net
    pdf = _canvas(path, "Ücret Bordrosu", GENERATOR_CREATORS["INCOME"])
    y = _header(pdf, "ÜCRET BORDROSU", f"{employer} — Dönem: {period}")
    y = _rows(
        pdf,
        y,
        [
            ("Ad Soyad", name),
            ("T.C. Kimlik No", tckn),
            ("İşveren", employer),
            ("Dönem", period),
            ("Brüt Ücret", money(gross)),
            ("SGK İşçi Payı", money(sgk)),
            ("İşsizlik Sigortası İşçi Payı", money(unemployment)),
            ("Gelir Vergisi", money(income_tax)),
            ("Damga Vergisi", money(stamp)),
            ("Toplam Kesinti", money(deductions)),
            ("Net Ücret", money(net_value)),
        ],
    )
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawString(60, y - 20, "Bu bordro elektronik ortamda üretilmiştir. (Sentetik demo belgesi)")
    pdf.save()
    return {"gross": gross, "deductions": deductions, "net": net_value, "true_net": true_net}


def make_bank_statement(
    path: Path,
    *,
    name: str,
    iban: str,
    transactions: list[dict],
    opening: float,
    closing: float | None = None,
) -> dict[str, float]:
    """Bank statement; ``closing`` can be forced to break the arithmetic."""
    true_closing = round(opening + sum(t["amount"] for t in transactions), 2)
    closing_value = true_closing if closing is None else closing
    pdf = _canvas(path, "Hesap Özeti", GENERATOR_CREATORS["BANK_STATEMENT"])
    y = _header(pdf, "HESAP ÖZETİ", "Anil Bank A.Ş. — Vadesiz TL Hesap")
    first = transactions[0]["date"] if transactions else ""
    last = transactions[-1]["date"] if transactions else ""
    y = _rows(
        pdf,
        y,
        [
            ("Hesap Sahibi", name),
            ("IBAN", iban),
            ("Dönem", f"{first} / {last}"),
            ("Açılış Bakiyesi", money(opening)),
        ],
    )
    pdf.setFont(FONT_BOLD, 9)
    pdf.drawString(60, y, "Tarih")
    pdf.drawString(130, y, "Açıklama")
    pdf.drawString(360, y, "Tutar")
    pdf.drawString(460, y, "Bakiye")
    y -= 14
    running = opening
    pdf.setFont(FONT_REGULAR, 8)
    for tx in transactions:
        running = round(running + tx["amount"], 2)
        if y < 70:
            pdf.showPage()
            pdf.setFont(FONT_REGULAR, 8)
            y = A4[1] - 60
        pdf.drawString(60, y, tx["date"])
        pdf.drawString(130, y, tx["description"][:38])
        pdf.drawRightString(430, y, money(tx["amount"]))
        pdf.drawRightString(540, y, money(running))
        y -= 12
    y -= 8
    pdf.setFont(FONT_BOLD, 10)
    pdf.drawString(60, max(y, 50), f"Kapanış Bakiyesi: {money(closing_value)}")
    pdf.save()
    return {"opening": opening, "closing": closing_value, "true_closing": true_closing}


def _qr(pdf: canvas.Canvas, value: str, x: float, y: float, size: float = 90) -> None:
    widget = QrCodeWidget(value)
    bounds = widget.getBounds()
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    drawing = Drawing(size, size, transform=[size / w, 0, 0, size / h, 0, 0])
    drawing.add(widget)
    renderPDF.draw(drawing, pdf, x, y)


def make_edevlet_document(
    path: Path,
    *,
    kind: str,
    name: str,
    tckn: str,
    detail: str,
    barcode: str | None = None,
) -> str:
    """e-Devlet barcoded document (``kind``: YERLESIM or SGK_HIZMET)."""
    barcode = barcode or make_barcode(f"{tckn}|{kind}")
    title = "YERLEŞİM YERİ VE DİĞER ADRES BELGESİ" if kind == "YERLESIM" else "SGK HİZMET DÖKÜMÜ"
    code = "ADDRESS" if kind == "YERLESIM" else "EMPLOYMENT"
    pdf = _canvas(path, title, GENERATOR_CREATORS[code])
    y = _header(pdf, title, "T.C. İçişleri Bakanlığı / SGK — e-Devlet Kapısı (sentetik)")
    rows = [("Ad Soyad", name), ("T.C. Kimlik No", tckn)]
    rows.append(("Adres", detail) if kind == "YERLESIM" else ("İşyeri", detail))
    rows.append(("Belge Tarihi", date(2026, 9, 1).isoformat()))
    rows.append(("Barkod No", barcode))
    y = _rows(pdf, y, rows)
    _qr(pdf, f"https://www.turkiye.gov.tr/belge-dogrulama?barkod={barcode}", 440, y - 60)
    pdf.setFont(FONT_REGULAR, 8)
    pdf.drawString(
        60,
        y - 30,
        "Belgenin doğruluğunu barkod numarası ile e-Devlet Kapısı'ndan sorgulayabilirsiniz.",
    )
    pdf.save()
    return barcode


def make_identity(path: Path, *, name: str, tckn: str, birth_date: str) -> None:
    pdf = _canvas(path, "Kimlik Kartı", GENERATOR_CREATORS["IDENTITY"])
    y = _header(pdf, "T.C. KİMLİK KARTI (ÖRNEK)", "Nüfus ve Vatandaşlık İşleri (sentetik)")
    _rows(pdf, y, [("Ad Soyad", name), ("T.C. Kimlik No", tckn), ("Doğum Tarihi", birth_date)])
    pdf.save()


def make_tax_plate(path: Path, *, name: str, tckn: str, tax_office: str, activity: str) -> None:
    pdf = _canvas(path, "Vergi Levhası", GENERATOR_CREATORS["TAX_PLATE"])
    y = _header(pdf, "VERGİ LEVHASI", "Gelir İdaresi Başkanlığı (sentetik)")
    _rows(
        pdf,
        y,
        [
            ("Adı Soyadı", name),
            ("Vergi Kimlik No", tckn),
            ("Vergi Dairesi", tax_office),
            ("Ana Faaliyet", activity),
        ],
    )
    pdf.save()


def tamper_pdf(source: Path, target: Path, *, overlays: list[tuple[str, str]]) -> None:
    """Overlay replacement text and save as an incremental update.

    ``overlays`` holds ``(old_text, new_text)`` pairs; the old value is covered
    through a redaction and the new value is written in Helvetica (a different
    font from the DejaVu body), then the file is saved incrementally with an
    editing-tool producer string and a later modification date.
    """
    import shutil

    import fitz

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    doc = fitz.open(str(target))
    for page in doc:
        for old, new in overlays:
            rects = page.search_for(old)
            for rect in rects:
                page.add_redact_annot(rect, fill=(1, 1, 1))
            if rects:
                page.apply_redactions()
            for rect in rects:
                page.insert_text((rect.x0, rect.y1 - 2), new, fontname="helv", fontsize=10)
    later = datetime.now(UTC) + timedelta(days=3)
    metadata = dict(doc.metadata or {})
    metadata["producer"] = "iLovePDF Online Editor"
    metadata["modDate"] = later.strftime("D:%Y%m%d%H%M%S+00'00'")
    doc.set_metadata(metadata)
    doc.saveIncr()
    doc.close()


def make_scanned_pdf(source: Path, target: Path, *, dpi: int = 200) -> Path:
    """Image-only copy of ``source`` (no text layer), as a scanner would produce."""
    import fitz

    target.parent.mkdir(parents=True, exist_ok=True)
    out = fitz.open()
    with fitz.open(str(source)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, pixmap=pix)
    out.set_metadata({"producer": "Scanner", "creator": "Scanner"})
    out.save(str(target))
    out.close()
    return target


def generate_applicant_bundle(
    directory: Path,
    *,
    name: str,
    tckn: str,
    iban: str,
    address: str,
    employer: str,
    net_income: float,
    transactions: list[dict],
    opening_balance: float,
    tamper_payslip: bool = False,
    tamper_statement: bool = False,
    self_employed: bool = False,
) -> dict[str, Path]:
    """Create a full document set for one applicant; returns code -> path."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    gross = round(net_income / 0.71, 2)
    make_identity(directory / "kimlik.pdf", name=name, tckn=tckn, birth_date="1988-04-12")
    paths["IDENTITY"] = directory / "kimlik.pdf"
    payslip = directory / "bordro.pdf"
    figures = make_payslip(
        payslip, name=name, tckn=tckn, employer=employer, period="2026-08", gross=gross
    )
    if tamper_payslip:
        inflated = round(figures["net"] * 1.8, 2)
        tampered = directory / "bordro_kurcalanmis.pdf"
        tamper_pdf(payslip, tampered, overlays=[(money(figures["net"]), money(inflated))])
        paths["INCOME"] = tampered
    else:
        paths["INCOME"] = payslip
    make_edevlet_document(
        directory / "sgk_hizmet.pdf", kind="SGK_HIZMET", name=name, tckn=tckn, detail=employer
    )
    paths["EMPLOYMENT"] = directory / "sgk_hizmet.pdf"
    make_edevlet_document(
        directory / "ikametgah.pdf", kind="YERLESIM", name=name, tckn=tckn, detail=address
    )
    paths["ADDRESS"] = directory / "ikametgah.pdf"
    recent = transactions[-45:]
    opening = (
        round(recent[0]["balance_after"] - recent[0]["amount"], 2) if recent else opening_balance
    )
    statement = directory / "hesap_ozeti.pdf"
    make_bank_statement(
        statement,
        name=name,
        iban=iban,
        transactions=recent,
        opening=opening,
        closing=(recent[-1]["balance_after"] + 25_000.0) if tamper_statement and recent else None,
    )
    paths["BANK_STATEMENT"] = statement
    if self_employed:
        make_tax_plate(
            directory / "vergi_levhasi.pdf",
            name=name,
            tckn=tckn,
            tax_office="Kadıköy",
            activity="Yazılım danışmanlığı",
        )
        paths["TAX_PLATE"] = directory / "vergi_levhasi.pdf"
    return paths
