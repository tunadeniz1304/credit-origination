"""Document tampering signals.

* Metadata (``pikepdf``): editing-tool producers, producer/creator mismatch,
  modification long after creation, and **incremental saves** (multiple
  ``%%EOF`` markers = content appended after issuance).
* Fonts (PyMuPDF spans): values rendered in a font foreign to the document.
* Arithmetic: payslip ``gross - deductions == net``; statement
  ``opening + sum(transactions) == closing``.
* e-Devlet barcode: presence + checksum verification through the (mock)
  e-Devlet verification endpoint; QR decoding with ``pyzbar`` when installed.

Each signal carries a weight; the document fraud score is their capped sum.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app.documents.extraction import (
    ExtractionResult,
    FieldValue,
    parse_amount,
    statement_transactions,
)
from app.documents.samples import GENERATOR_CREATORS, verify_barcode

EDITING_TOOLS = ("ilovepdf", "smallpdf", "word", "photoshop", "sejda", "pdfescape", "foxit phantom")
WEIGHTS = {
    "incremental_update": 0.35,
    "editing_tool": 0.30,
    "producer_mismatch": 0.15,
    "modified_after_creation": 0.15,
    "font_inconsistency": 0.20,
    "arithmetic_mismatch": 0.45,
    "barcode_invalid": 0.50,
    "barcode_missing": 0.25,
}
LABELS = {
    "incremental_update": "PDF sonradan artımlı olarak kaydedilmiş (düzenleme izi)",
    "editing_tool": "Belge bir PDF düzenleme aracıyla işlenmiş",
    "producer_mismatch": "Üretici yazılım beklenen kaynakla uyumsuz",
    "modified_after_creation": "Değişiklik tarihi oluşturma tarihinden çok sonra",
    "font_inconsistency": "Tutar alanlarında belgeye yabancı yazı tipi",
    "arithmetic_mismatch": "Belgedeki tutarlar aritmetik olarak tutarsız",
    "barcode_invalid": "e-Devlet barkodu doğrulanamadı",
    "barcode_missing": "e-Devlet belgesinde barkod bulunamadı",
}


class FraudSignal(BaseModel):
    code: str
    label: str
    weight: float
    detail: str = ""


class FraudReport(BaseModel):
    score: float
    signals: list[FraudSignal]


def _signal(code: str, detail: str = "") -> FraudSignal:
    return FraudSignal(code=code, label=LABELS[code], weight=WEIGHTS[code], detail=detail)


def _pdf_date(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.match(r"D:(\d{14})", str(value))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def metadata_signals(path: Path, code: str) -> list[FraudSignal]:
    import pikepdf

    signals: list[FraudSignal] = []
    raw = path.read_bytes()
    eof_markers = raw.count(b"%%EOF")
    if eof_markers > 1:
        signals.append(_signal("incremental_update", f"{eof_markers} revizyon"))
    try:
        with pikepdf.open(str(path)) as pdf:
            info = pdf.docinfo
            producer = str(info.get("/Producer", "") or "")
            creator = str(info.get("/Creator", "") or "")
            created = _pdf_date(str(info.get("/CreationDate", "") or ""))
            modified = _pdf_date(str(info.get("/ModDate", "") or ""))
    except Exception:  # corrupt/encrypted PDFs are handled by the caller
        return signals
    lowered = producer.lower()
    if any(tool in lowered for tool in EDITING_TOOLS):
        signals.append(_signal("editing_tool", producer))
    expected = GENERATOR_CREATORS.get(code)
    if expected and creator == expected and "reportlab" not in lowered:
        signals.append(_signal("producer_mismatch", f"creator={creator} producer={producer}"))
    if created and modified and (modified - created).total_seconds() > 86_400:
        signals.append(
            _signal("modified_after_creation", f"{created:%Y-%m-%d} → {modified:%Y-%m-%d}")
        )
    return signals


def font_signals(path: Path) -> list[FraudSignal]:
    import fitz

    fonts_by_kind: dict[str, set[str]] = {"amount": set(), "text": set()}
    with fitz.open(str(path)) as doc:
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        base = re.sub(r"^[A-Z]{6}\+", "", span.get("font", ""))
                        family = base.split("-")[0].lower()
                        kind = "amount" if re.search(r"\d+,\d{2}", text) else "text"
                        fonts_by_kind[kind].add(family)
    foreign = fonts_by_kind["amount"] - fonts_by_kind["text"]
    if foreign:
        return [_signal("font_inconsistency", ", ".join(sorted(foreign)))]
    return []


def _field(fields: list[FieldValue], name: str) -> float | None:
    for f in fields:
        if f.name == name:
            try:
                return parse_amount(f.value)
            except ValueError:
                return None
    return None


def arithmetic_signals(
    code: str, fields: list[FieldValue], text: ExtractionResult
) -> list[FraudSignal]:
    if code == "INCOME":
        gross, deductions, net = (
            _field(fields, "brut_ucret"),
            _field(fields, "toplam_kesinti"),
            _field(fields, "net_ucret"),
        )
        if None not in (gross, deductions, net) and abs(gross - deductions - net) > 1.0:  # type: ignore[operator]
            return [
                _signal(
                    "arithmetic_mismatch",
                    f"brüt−kesinti={gross - deductions:,.2f} ≠ net={net:,.2f}",
                )
            ]  # type: ignore[operator]
    if code == "BANK_STATEMENT":
        opening, closing = _field(fields, "acilis_bakiyesi"), _field(fields, "kapanis_bakiyesi")
        movements = statement_transactions(text)
        if opening is not None and closing is not None and movements:
            expected = round(opening + sum(movements), 2)
            if abs(expected - closing) > 1.0:
                return [
                    _signal(
                        "arithmetic_mismatch",
                        f"açılış+hareketler={expected:,.2f} ≠ kapanış={closing:,.2f}",
                    )
                ]
    return []


def barcode_signals(code: str, fields: list[FieldValue], path: Path) -> list[FraudSignal]:
    if code not in ("ADDRESS", "EMPLOYMENT"):
        return []
    barcode = next((f.value for f in fields if f.name == "barkod"), None)
    if not barcode:
        return [_signal("barcode_missing")]
    signals: list[FraudSignal] = []
    if not verify_barcode(barcode):
        signals.append(_signal("barcode_invalid", barcode))
    decoded = _decode_qr(path)
    if decoded is not None and barcode not in decoded:
        signals.append(_signal("barcode_invalid", "QR içeriği barkodla uyuşmuyor"))
    return signals


def _decode_qr(path: Path) -> str | None:
    try:
        from pyzbar.pyzbar import decode  # optional extra
    except ImportError:
        return None
    try:  # pragma: no cover - optional extra
        import fitz
        from PIL import Image

        with fitz.open(str(path)) as doc:
            pix = doc[0].get_pixmap(dpi=150)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        results = decode(image)
        return " ".join(r.data.decode("utf-8", "ignore") for r in results) if results else ""
    except Exception:  # pragma: no cover
        return None


def analyse_document(
    path: Path, code: str, mime: str, fields: list[FieldValue], text: ExtractionResult
) -> FraudReport:
    signals: list[FraudSignal] = []
    if mime == "application/pdf":
        signals += metadata_signals(path, code)
        signals += font_signals(path)
    signals += arithmetic_signals(code, fields, text)
    signals += barcode_signals(code, fields, path)
    score = min(1.0, sum(s.weight for s in signals))
    return FraudReport(score=round(score, 3), signals=signals)
