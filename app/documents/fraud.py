"""Document tampering signals — independent of who produced the file.

v1 flagged ``producer_mismatch`` when a document did not come from *this
project's own* generator (reportlab), so it only caught its own forgeries and
flagged genuine files from any other tool. The signals are now generic
(``rules/fraud.yaml``):

* **Producer profile** per document type: known PDF editors / image editors are
  suspicious everywhere; word processors are suspicious for system-issued
  documents (payslips, statements, e-Devlet). An unknown producer alone is
  never a signal.
* **Dates**: ``ModDate`` later than ``CreationDate`` by more than a threshold.
* **Incremental saves**: more than one revision (``%%EOF`` / ``startxref``).
* **XMP vs Info**: producer, creator tool or modification date disagree.
* **Fonts**: amounts in a font foreign to the body text, amount digits in a
  different font or size, too many font families on one page.
* **Layers**: text drawn over white filled shapes (white-out), or a text layer
  patched on top of a scanned page image.
* **Arithmetic** (payslip ``gross − deductions = net``; statement
  ``opening + Σ movements = closing``) and **e-Devlet barcode** checks.

Each signal carries a configured weight; the document score is their capped sum.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.rules import FraudConfig, load_fraud
from app.documents.extraction import (
    ExtractionResult,
    FieldValue,
    parse_amount,
    statement_transactions,
)
from app.documents.samples import verify_barcode

AMOUNT_RE = re.compile(r"\d+,\d{2}")
WHITE_FILL = 0.97  # a fill this close to white counts as white-out
LABELS = {
    "incremental_update": "PDF sonradan artımlı olarak kaydedilmiş (düzenleme izi)",
    "editing_tool": "Belge bir PDF düzenleme/görüntü aracıyla işlenmiş",
    "producer_unexpected": (
        "Sistem tarafından üretilmesi gereken belge ofis/kelime işlemci aracından çıkmış"
    ),
    "modified_after_creation": "Değişiklik tarihi oluşturma tarihinden çok sonra",
    "metadata_inconsistent": "XMP üst verisi ile belge bilgi sözlüğü çelişiyor",
    "font_inconsistency": "Tutar alanlarında belgeye yabancı yazı tipi",
    "font_mix": "Tek sayfada olağan dışı sayıda yazı tipi ailesi",
    "digit_font_mismatch": "Tutar rakamları farklı yazı tipi veya boyutta",
    "whiteout_overlay": "Beyaz dolgulu alanın üzerine sonradan metin yazılmış",
    "text_over_image": "Taranmış sayfa görüntüsünün üzerine metin katmanı eklenmiş",
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


def _signal(code: str, detail: str = "", cfg: FraudConfig | None = None) -> FraudSignal:
    cfg = cfg or load_fraud()
    return FraudSignal(code=code, label=LABELS[code], weight=cfg.weights[code], detail=detail)


def _pdf_date(value: Any) -> datetime | None:
    """PDF (``D:20260901120000``) or XMP (``2026-09-01T12:00:00``) date → naive datetime."""
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    if len(digits) < 14:
        return None
    try:
        return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _matches(value: str, patterns: list[str]) -> str | None:
    lowered = value.lower()
    return next((p for p in patterns if p in lowered), None)


def revision_count(raw: bytes) -> int:
    return max(raw.count(b"%%EOF"), raw.count(b"startxref"))


def metadata_signals(path: Path, code: str, cfg: FraudConfig | None = None) -> list[FraudSignal]:
    import pikepdf

    cfg = cfg or load_fraud()
    signals: list[FraudSignal] = []
    revisions = revision_count(path.read_bytes())
    if revisions > 1:
        signals.append(_signal("incremental_update", f"{revisions} revizyon", cfg))
    try:
        with pikepdf.open(str(path)) as pdf:
            info = pdf.docinfo
            producer = str(info.get("/Producer", "") or "")
            creator = str(info.get("/Creator", "") or "")
            created = _pdf_date(info.get("/CreationDate"))
            modified = _pdf_date(info.get("/ModDate"))
            meta = pdf.open_metadata()
            xmp_producer = str(meta.get("pdf:Producer", "") or "")
            xmp_creator = str(meta.get("xmp:CreatorTool", "") or "")
            xmp_modified = _pdf_date(meta.get("xmp:ModifyDate"))
    except Exception:  # corrupt/encrypted PDFs are handled by the caller
        return signals
    tools = " | ".join(t for t in (producer, creator) if t)
    editor = _matches(tools, cfg.editing_tools)
    if editor:
        signals.append(_signal("editing_tool", tools, cfg))
    profile = cfg.profiles.get(code)
    if profile and not editor and _matches(tools, profile.unexpected):
        signals.append(_signal("producer_unexpected", tools, cfg))
    limit = cfg.thresholds.modified_after_creation_hours * 3600
    if created and modified and (modified - created).total_seconds() > limit:
        signals.append(
            _signal("modified_after_creation", f"{created:%Y-%m-%d} → {modified:%Y-%m-%d}", cfg)
        )
    conflicts = []
    if xmp_producer and producer and xmp_producer != producer:
        conflicts.append(f"Producer: XMP='{xmp_producer}' Info='{producer}'")
    if xmp_creator and creator and xmp_creator != creator:
        conflicts.append(f"Creator: XMP='{xmp_creator}' Info='{creator}'")
    if xmp_modified and modified and abs((xmp_modified - modified).total_seconds()) > limit:
        conflicts.append("ModDate farklı")
    if conflicts:
        signals.append(_signal("metadata_inconsistent", "; ".join(conflicts), cfg))
    return signals


def _family(font: str) -> str:
    """``ABCDEF+DejaVuSans-Bold`` → ``dejavusans`` (subset prefix and style dropped)."""
    base = re.sub(r"^[A-Z]{6}\+", "", font)
    return base.split("-")[0].split(",")[0].lower()


def _overlap(inner: Any, outer: Any) -> float:
    """Share of ``inner`` covered by ``outer``."""
    import fitz

    a, b = fitz.Rect(inner), fitz.Rect(outer)
    inter = a & b
    area = a.get_area()
    return inter.get_area() / area if area and not inter.is_empty else 0.0


def font_signals(path: Path, cfg: FraudConfig | None = None) -> list[FraudSignal]:
    """Font families and sizes of amounts vs body text; white-out and image patches."""
    import fitz

    cfg = cfg or load_fraud()
    th = cfg.thresholds
    fonts_by_kind: dict[str, set[str]] = {"amount": set(), "text": set()}
    amount_styles: dict[tuple[str, int], int] = {}
    families_per_page: list[int] = []
    whiteouts: list[str] = []
    patched = 0
    with fitz.open(str(path)) as doc:
        for page in doc:
            families: set[str] = set()
            white = [
                d["rect"]
                for d in page.get_drawings()
                if d.get("fill") is not None
                and min(d["fill"]) >= WHITE_FILL
                and d["rect"].get_area() > 0
            ]
            page_area = page.rect.get_area()
            images = [
                fitz.Rect(img["bbox"])
                for img in page.get_image_info()
                if fitz.Rect(img["bbox"]).get_area() >= th.image_page_coverage * page_area
            ]
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        family = _family(span.get("font", ""))
                        families.add(family)
                        is_amount = bool(AMOUNT_RE.search(text))
                        fonts_by_kind["amount" if is_amount else "text"].add(family)
                        if is_amount:
                            size = round(float(span.get("size", 0)) / th.digit_size_tolerance_pt)
                            amount_styles[(family, size)] = amount_styles.get((family, size), 0) + 1
                        bbox = span.get("bbox")
                        if any(_overlap(bbox, w) >= th.whiteout_min_overlap for w in white):
                            whiteouts.append(text)
                        if any(_overlap(bbox, im) >= 1.0 for im in images):
                            patched += 1
            families_per_page.append(len(families))
    signals: list[FraudSignal] = []
    foreign = fonts_by_kind["amount"] - fonts_by_kind["text"]
    if foreign:
        signals.append(_signal("font_inconsistency", ", ".join(sorted(foreign)), cfg))
    # Same size, different family: an amount re-typed in another font. Size-only
    # differences (headers, totals, table rows) are ordinary layout.
    families_by_size: dict[int, set[str]] = {}
    for family, size in amount_styles:
        families_by_size.setdefault(size, set()).add(family)
    mixed = {size: fams for size, fams in families_by_size.items() if len(fams) > 1}
    if mixed:
        detail = "; ".join(
            f"{size * th.digit_size_tolerance_pt:.1f}pt: {', '.join(sorted(fams))}"
            for size, fams in sorted(mixed.items())
        )
        signals.append(_signal("digit_font_mismatch", detail, cfg))
    if families_per_page and max(families_per_page) > th.max_font_families_per_page:
        signals.append(_signal("font_mix", f"{max(families_per_page)} aile", cfg))
    if whiteouts:
        signals.append(_signal("whiteout_overlay", ", ".join(whiteouts[:3]), cfg))
    if patched:
        signals.append(_signal("text_over_image", f"{patched} metin parçası", cfg))
    return signals


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
        if (
            gross is not None
            and deductions is not None
            and net is not None
            and abs(gross - deductions - net) > 1.0
        ):
            return [
                _signal(
                    "arithmetic_mismatch",
                    f"brüt−kesinti={gross - deductions:,.2f} ≠ net={net:,.2f}",
                )
            ]
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
    cfg = load_fraud()
    signals: list[FraudSignal] = []
    if mime == "application/pdf":
        signals += metadata_signals(path, code, cfg)
        signals += font_signals(path, cfg)
    signals += arithmetic_signals(code, fields, text)
    signals += barcode_signals(code, fields, path)
    score = min(1.0, sum(s.weight for s in signals))
    return FraudReport(score=round(score, 3), signals=signals)
