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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

TCKN_RE = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
IBAN_RE = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?90[\s-]?)?0?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


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
        if value in self._forward:
            return self._forward[value]
        token = self._pseudonym(kind, value)
        self._forward[value] = token
        self._reverse[token] = value
        return token

    def redact(self, text: str) -> str:
        # Longest values first so "Ali Yılmaz" wins over "Ali".
        for value in sorted(self._forward, key=len, reverse=True):
            text = re.sub(re.escape(value), self._forward[value], text, flags=re.IGNORECASE)
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
