"""ReportLab font registration with embedded DejaVu Sans (Turkish glyphs).

The built-in Helvetica lacks proper glyphs/encoding for ı, ş, ğ and İ, which
corrupted Turkish text in the old reports. Every PDF the platform produces
embeds DejaVu Sans instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_REGULAR = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"


@lru_cache(maxsize=1)
def register_fonts() -> tuple[str, str]:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    from reportlab.lib.fonts import addMapping

    addMapping(FONT_REGULAR, 0, 0, FONT_REGULAR)
    addMapping(FONT_REGULAR, 1, 0, FONT_BOLD)
    addMapping(FONT_REGULAR, 0, 1, FONT_REGULAR)
    addMapping(FONT_REGULAR, 1, 1, FONT_BOLD)
    return FONT_REGULAR, FONT_BOLD
