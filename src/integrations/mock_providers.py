"""Mock external data providers (KKB credit bureau, e-Devlet).

Deterministic and fully offline: every derived value is a pure function of the
applicant's ``identity_no`` so the same applicant always yields the same data,
across runs and machines.

.. note:: The derived values MUST NOT use Python's builtin ``hash()`` — it is
   is salted per process, so ``hash("...")`` changes on every run. A stable SHA-256
   digest is used instead.

Sample ``identity_no`` cells used by the demo/test flow (see ``src/main.py`` and
``tests/test_flow.py``) — derived values verified by running the exact score
function:

- ``12345678901`` -> KBB 1450, debt 134070 -> passes ``min_kbb_score`` (complete)
- ``34567890123`` -> KBB  619, debt 176116 -> fails ``min_kbb_score`` (high risk)
"""
from __future__ import annotations

import hashlib
import random
import time

from src.agents.base import load_config
from src.logger import get_logger
from src.models import EmploymentRecord, KBBReport

_EMPLOYERS = (
    "Yıldız Holding A.Ş.",
    "Anadolu Bilişim Ltd.",
    "Ege Lojistik San. Tic.",
    "KARBON Enerji A.Ş.",
    "Bereket Gıda Pazarlama",
)

_RISK_CLASSES = (
    (1500, "ÇOK DÜŞÜK RİSK"),
    (1100, "DÜŞÜK RİSK"),
    (700, "ORTA RİSK"),
    (0, "YÜKSEK RİSK"),
)

_KBB_MAX_SCORE = 1900
_KBB_MIN_SCORE = 300
_DEBT_MIN = 5_000.0
_DEBT_MAX = 180_000.0


def _stable_digest(value: str) -> int:
    """Stable, process-independent 64-bit digest of a string."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _range_value(value: str, lo: int, hi: int) -> int:
    """Deterministic integer in ``[lo, hi)`` per (value, lo, hi)."""
    return lo + _stable_digest(f"{value}|{lo}|{hi}") % (hi - lo)


def _sleep_for(config: dict) -> None:
    """Simulate network latency from ``latency_ms=[min, max]`` config."""
    latency_ms = config.get("latency_ms", [20, 120])
    lo, hi = latency_ms[0], latency_ms[1]
    if hi <= lo:
        hi = lo + 1
    time.sleep(random.uniform(lo, hi) / 1000.0)


class KKBClient:
    """Mock credit bureau client; ``get_report`` simulates KKB HTTP latency."""

    SERVICE = "kkb"

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or load_config()
        self._config = cfg["api"][self.SERVICE]
        self.logger = get_logger("api.kkb")
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def get_report(self, identity_no: str) -> KBBReport:
        self.logger.info("KKB report requested for %s", identity_no)
        _sleep_for(self._config)
        self._calls += 1

        score = _range_value(identity_no, _KBB_MIN_SCORE, _KBB_MAX_SCORE + 1)
        total_debt = _range_value(identity_no, int(_DEBT_MIN), int(_DEBT_MAX) + 1)
        risk_class = next(label for threshold, label in _RISK_CLASSES if score >= threshold)
        report = KBBReport(score=score, risk_class=risk_class, total_debt=float(total_debt))
        self.logger.info(
            "KKB report delivered: score=%d risk='%s' total_debt=%.2f", report.score, report.risk_class,
            report.total_debt,
        )
        return report


class EDevletClient:
    """Mock e-Devlet employment-record client."""

    SERVICE = "edevlet"

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or load_config()
        self._config = cfg["api"][self.SERVICE]
        self.logger = get_logger("api.edevlet")
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def get_employment(self, identity_no: str) -> EmploymentRecord:
        self.logger.info("e-Devlet employment record requested for %s", identity_no)
        _sleep_for(self._config)
        self._calls += 1

        index = _stable_digest(identity_no) % len(_EMPLOYERS)
        employer = _EMPLOYERS[index]
        years = _range_value(identity_no, 0, 35)
        verified = _range_value(identity_no, 0, 10) != 0  # ~9/10 verified
        record = EmploymentRecord(employer=employer, years=years, verified=verified)
        self.logger.info(
            "Employment record delivered: employer='%s' years=%d verified=%s", employer, years, verified
        )
        return record
