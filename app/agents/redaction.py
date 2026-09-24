"""KVKK redaction: pseudonymise PII before any text leaves for an LLM.

A :class:`Redactor` is seeded with the applicant's known identifiers (name,
TCKN, phone, IBAN, address, e-mail) and additionally catches unknown PII by
pattern. Every value is replaced with a stable pseudonym such as
``BASVURAN_1`` or ``IBAN_…1234``; :meth:`Redactor.restore` maps pseudonyms back
in the model output. :class:`PIIMaskingFilter` applies pattern masking to log
records so PII never reaches log files in clear text.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

TCKN_RE = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
IBAN_RE = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?90[\s-]?)?0?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


# Turkish-aware folding. Python's case-insensitive matching treats "İ"/"ı" as
# unrelated to "I"/"i", and core-banking systems often ASCII-fold names
# ("IŞIK" → "ISIK"), so every letter matches all its Turkish/ASCII variants.
_FOLD = str.maketrans(
    {"ı": "i", "İ": "i", "I": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u"}
    | {"ö": "o", "Ö": "o", "ç": "c", "Ç": "c", "â": "a", "Â": "a", "î": "i", "Î": "i"}
    | {"û": "u", "Û": "u"}
)
_VARIANTS: dict[str, str] = {}
for _char in "ıİIişŞsSğĞgGüÜuUöÖoOçÇcCâÂaAîÎûÛ":
    _key = _char.translate(_FOLD).lower()
    _VARIANTS[_key] = _VARIANTS.get(_key, "") + _char


def turkish_fold(text: str) -> str:
    """Case- and diacritic-insensitive key for Turkish text (``"İSMAİL IŞIK"`` → ``"ismail isik"``)."""
    return unicodedata.normalize("NFC", text).translate(_FOLD).lower()


def turkish_pattern(value: str) -> re.Pattern[str]:
    """Regex matching ``value`` in any Turkish/ASCII casing (whitespace-tolerant)."""
    parts: list[str] = []
    for char in turkish_fold(value):
        if char.isspace():
            parts.append(r"\s+")
        elif char in _VARIANTS:
            parts.append("[" + re.escape(_VARIANTS[char] + char + char.upper()) + "]")
        else:
            parts.append(re.escape(char))
    # Leading boundary only: "Ali" must not match inside "Mali", but suffixed forms
    # ("İsmailin", "Işık'a") are still masked (privacy first).
    boundary = r"(?<![\w])" if value[:1].isalnum() else ""
    return re.compile(boundary + "".join(parts), re.IGNORECASE)


def mask_text(text: str) -> str:
    """Pattern-only masking used for logs (no restore map)."""
    text = IBAN_RE.sub(lambda m: f"IBAN_…{_digits(m.group(0))[-4:]}", text)
    text = EMAIL_RE.sub("EPOSTA_***", text)
    text = PHONE_RE.sub(lambda m: f"TEL_…{_digits(m.group(0))[-2:]}", text)
    text = TCKN_RE.sub(lambda m: f"TCKN_…{m.group(0)[-2:]}", text)
    return text


@dataclass
class Redactor:
    """Reversible pseudonymisation for one LLM conversation."""

    _forward: dict[str, str] = field(default_factory=dict)
    _reverse: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)
    _keys: dict[str, str] = field(default_factory=dict)  # folded value -> registered value

    @classmethod
    def for_applicant(
        cls,
        *,
        name: str | None = None,
        identity_no: str | None = None,
        phone: str | None = None,
        iban: str | None = None,
        address: str | None = None,
        email: str | None = None,
        extra_names: Iterable[str] = (),
    ) -> Redactor:
        redactor = cls()
        if name:
            redactor.register(name, "BASVURAN")
            for part in name.split():
                if len(part) > 2:
                    redactor.register(part, "BASVURAN_AD")
        for other in extra_names:
            redactor.register(other, "KISI")
        if identity_no:
            redactor.register(identity_no, "TCKN")
        if phone:
            redactor.register(phone, "TEL")
        if iban:
            redactor.register(iban, "IBAN")
        if address:
            redactor.register(address, "ADRES")
        if email:
            redactor.register(email, "EPOSTA")
        return redactor

    def _pseudonym(self, kind: str, value: str) -> str:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        index = self._counters[kind]
        if kind == "IBAN":
            return f"IBAN_…{_digits(value)[-4:]}"
        if kind == "TEL":
            return f"TEL_{index}_…{_digits(value)[-2:]}"
        return f"{kind}_{index}"

    def register(self, value: str, kind: str) -> str:
        value = value.strip()
        if not value:
            return value
        key = turkish_fold(value)
        if key in self._keys:
            return self._forward[self._keys[key]]
        token = self._pseudonym(kind, value)
        self._forward[value] = token
        self._reverse[token] = value
        self._keys[key] = value
        return token

    def redact(self, text: str) -> str:
        # Longest values first so "Ali Yılmaz" wins over "Ali"; matching is Turkish-aware
        # (İ/ı/I/i and ASCII-folded spellings).
        for value in sorted(self._forward, key=len, reverse=True):
            text = turkish_pattern(value).sub(self._forward[value], text)
        text = IBAN_RE.sub(lambda m: self.register(m.group(0), "IBAN"), text)
        text = EMAIL_RE.sub(lambda m: self.register(m.group(0), "EPOSTA"), text)
        text = PHONE_RE.sub(lambda m: self.register(m.group(0), "TEL"), text)
        text = TCKN_RE.sub(lambda m: self.register(m.group(0), "TCKN"), text)
        return text

    def redact_obj(self, obj: Any) -> Any:
        """Recursively redact strings in JSON-like structures."""
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, list):
            return [self.redact_obj(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self.redact_obj(value) for key, value in obj.items()}
        return obj

    def restore(self, text: str) -> str:
        for token in sorted(self._reverse, key=len, reverse=True):
            text = text.replace(token, self._reverse[token])
        return text

    @property
    def mapping(self) -> dict[str, str]:
        """Pseudonym -> original (for tests; never log this)."""
        return dict(self._reverse)


class PIIMaskingFilter(logging.Filter):
    """Masks TCKN / IBAN / phone / e-mail in every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        masked = mask_text(message)
        if masked != message:
            record.msg = masked
            record.args = None
        return True
