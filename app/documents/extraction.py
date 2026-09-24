"""Text and field extraction (IDP).

Text: PyMuPDF for digital PDFs; when a page has no text layer (scans,
images) OCR is attempted with ``pytesseract`` + the ``tur`` model if the
optional extra is installed, otherwise the document is marked
``OCR_GEREKLI`` and routed to a specialist. Fields: label-anchored regular
expressions per document type, with a confidence score and the source page +
bounding box (for the reviewer's document viewer).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page: int
    text: str


class ExtractionResult(BaseModel):
    pages: list[PageText] = Field(default_factory=list)
    source: str = "none"  # digital | ocr | none
    ocr_available: bool = False

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


class FieldValue(BaseModel):
    name: str
    value: str
    confidence: float
    page: int | None = None
    bbox: list[float] | None = None
    source: str = "regex"


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def extract_text(path: Path, mime: str) -> ExtractionResult:
    ocr = _ocr_available()
    if mime == "text/plain":
        return ExtractionResult(
            pages=[PageText(page=1, text=path.read_text(encoding="utf-8", errors="replace"))],
            source="digital",
            ocr_available=ocr,
        )
    if mime == "application/pdf":
        import fitz

        pages: list[PageText] = []
        with fitz.open(str(path)) as doc:
            for index, page in enumerate(doc, start=1):
                pages.append(PageText(page=index, text=page.get_text("text", sort=True)))
        if any(p.text.strip() for p in pages):
            return ExtractionResult(pages=pages, source="digital", ocr_available=ocr)
    if not ocr:
        return ExtractionResult(source="none", ocr_available=False)
    return _ocr(path, mime)  # pragma: no cover - optional extra


def _ocr(path: Path, mime: str) -> ExtractionResult:  # pragma: no cover - optional extra
    import pytesseract
    from PIL import Image

    images = []
    if mime == "application/pdf":
        import fitz

        with fitz.open(str(path)) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    else:
        images.append(Image.open(path))
    pages = [
        PageText(page=i, text=pytesseract.image_to_string(img, lang="tur"))
        for i, img in enumerate(images, start=1)
    ]
    return ExtractionResult(pages=pages, source="ocr", ocr_available=True)


_AMOUNT = r"(-?\d{1,3}(?:\.\d{3})*,\d{2})\s*TL"
FIELD_PATTERNS: dict[str, dict[str, str]] = {
    "INCOME": {
        "ad_soyad": r"Ad Soyad:\s*(.+)",
        "tckn": r"T\.C\. Kimlik No:\s*(\d{11})",
        "isveren": r"İşveren:\s*(.+)",
        "donem": r"Dönem:\s*([\d-]{7})",
        "brut_ucret": r"Brüt Ücret:\s*" + _AMOUNT,
        "toplam_kesinti": r"Toplam Kesinti:\s*" + _AMOUNT,
        "net_ucret": r"Net Ücret:\s*" + _AMOUNT,
    },
    "BANK_STATEMENT": {
        "hesap_sahibi": r"Hesap Sahibi:\s*(.+)",
        "iban": r"IBAN:\s*(TR[\d ]{24,32})",
        "acilis_bakiyesi": r"Açılış Bakiyesi:\s*" + _AMOUNT,
        "kapanis_bakiyesi": r"Kapanış Bakiyesi:\s*" + _AMOUNT,
    },
    "ADDRESS": {
        "ad_soyad": r"Ad Soyad:\s*(.+)",
        "tckn": r"T\.C\. Kimlik No:\s*(\d{11})",
        "adres": r"Adres:\s*(.+)",
        "barkod": r"Barkod No:\s*(ED\d{14})",
    },
    "EMPLOYMENT": {
        "ad_soyad": r"Ad Soyad:\s*(.+)",
        "tckn": r"T\.C\. Kimlik No:\s*(\d{11})",
        "isyeri": r"İşyeri:\s*(.+)",
        "barkod": r"Barkod No:\s*(ED\d{14})",
    },
    "IDENTITY": {
        "ad_soyad": r"Ad Soyad:\s*(.+)",
        "tckn": r"T\.C\. Kimlik No:\s*(\d{11})",
        "dogum_tarihi": r"Doğum Tarihi:\s*([\d-]{10})",
    },
    "TAX_PLATE": {
        "ad_soyad": r"Adı Soyadı:\s*(.+)",
        "vkn": r"Vergi Kimlik No:\s*(\d{10,11})",
        "vergi_dairesi": r"Vergi Dairesi:\s*(.+)",
    },
}


def parse_amount(text: str) -> float:
    """``45.000,00`` -> ``45000.0``."""
    return float(text.replace(".", "").replace(",", "."))


def _bbox(path: Path, page_no: int, value: str) -> list[float] | None:
    try:
        import fitz

        with fitz.open(str(path)) as doc:
            rects = doc[page_no - 1].search_for(value)
            if rects:
                r = rects[0]
                return [round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)]
    except Exception:
        return None
    return None


def extract_fields(
    code: str, result: ExtractionResult, path: Path | None = None
) -> list[FieldValue]:
    patterns = FIELD_PATTERNS.get(code, {})
    base_conf = 0.95 if result.source == "digital" else 0.65
    fields: list[FieldValue] = []
    for name, pattern in patterns.items():
        matches: list[tuple[int, str]] = []
        for page in result.pages:
            for match in re.finditer(pattern, page.text):
                matches.append((page.page, match.group(1).strip()))
        if not matches:
            continue
        values = {value for _, value in matches}
        confidence = base_conf if len(values) == 1 else base_conf * 0.6
        page_no, value = matches[0] if name != "kapanis_bakiyesi" else matches[-1]
        bbox = _bbox(path, page_no, value) if path and result.source == "digital" else None
        fields.append(
            FieldValue(
                name=name, value=value, confidence=round(confidence, 3), page=page_no, bbox=bbox
            )
        )
    return fields


def statement_transactions(result: ExtractionResult) -> list[float]:
    """Transaction amounts listed on a statement (for the arithmetic check)."""
    row = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+.*?" + _AMOUNT + r"\s+" + _AMOUNT + r"\s*$")
    amounts: list[float] = []
    for page in result.pages:
        for line in page.text.splitlines():
            match = row.match(line)
            if match:
                amounts.append(parse_amount(match.group(1)))
    return amounts
