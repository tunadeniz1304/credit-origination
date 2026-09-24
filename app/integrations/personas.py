"""Persona-driven, internally consistent mock data (KKB, SGK, GİB, open banking).

Each identity maps deterministically (SHA-256 of the TCKN, never Python's
salted ``hash``) to a persona that drives every provider coherently: a
"clean" salaried applicant has a high bureau score, no arrears, regular
salary deposits matching the declared income and moderate spending, while an
"over-indebted" one shows high utilisation, many instalments and thin
balances. Demo identities are pinned to named personas. All data is
synthetic and every endpoint is labelled as a mock.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date, timedelta

from app.kyc.tckn import synthetic_tckn


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    bureau_hit: bool = True
    score_range: tuple[int, int] = (1450, 1750)
    active_loans: tuple[int, int] = (0, 2)
    existing_dsr: tuple[float, float] = (0.05, 0.15)  # existing instalments / income
    utilisation: tuple[float, float] = (0.10, 0.40)
    delinquencies: tuple[int, int] = (0, 0)
    max_dpd: tuple[int, int] = (0, 0)
    inquiries: tuple[int, int] = (0, 2)
    legal_followup: bool = False
    employment_months: tuple[int, int] = (36, 180)
    income_truth: float = 1.0  # bank-observed salary / declared income
    income_noise: float = 0.03  # month-to-month salary variation
    spend_ratio: tuple[float, float] = (0.55, 0.75)
    gambling_share: float = 0.0
    nsf_events: tuple[int, int] = (0, 0)
    negative_days: tuple[int, int] = (0, 0)
    self_employed: bool = False


PERSONAS: dict[str, Persona] = {
    "temiz": Persona("temiz", "Temiz maaşlı"),
    "ince_dosya": Persona(
        "ince_dosya",
        "İnce dosya (kredi geçmişi yok)",
        bureau_hit=False,
        score_range=(0, 0),
        active_loans=(0, 0),
        existing_dsr=(0.0, 0.0),
        utilisation=(0.0, 0.0),
        inquiries=(0, 1),
        employment_months=(24, 60),
        spend_ratio=(0.45, 0.60),
    ),
    "asiri_borclu": Persona(
        "asiri_borclu",
        "Aşırı borçlu",
        score_range=(1180, 1350),
        active_loans=(3, 5),
        existing_dsr=(0.22, 0.30),
        utilisation=(0.80, 0.95),
        inquiries=(3, 6),
        spend_ratio=(0.85, 0.98),
        negative_days=(5, 20),
    ),
    "yuksek_dsr": Persona(
        "yuksek_dsr",
        "Yüksek borç servis oranı",
        score_range=(1480, 1650),
        active_loans=(1, 2),
        existing_dsr=(0.16, 0.20),
        utilisation=(0.30, 0.45),
        inquiries=(1, 2),
        employment_months=(48, 120),
        spend_ratio=(0.60, 0.72),
    ),
    "gecikmeli": Persona(
        "gecikmeli",
        "Gecikme geçmişli",
        score_range=(720, 980),
        active_loans=(2, 4),
        existing_dsr=(0.20, 0.32),
        utilisation=(0.75, 0.95),
        delinquencies=(2, 5),
        max_dpd=(30, 89),
        inquiries=(4, 8),
        income_noise=0.18,
        spend_ratio=(0.90, 1.05),
        nsf_events=(1, 4),
        negative_days=(20, 60),
        gambling_share=0.04,
        employment_months=(6, 30),
    ),
    "gri": Persona(
        "gri",
        "Gri bölge",
        score_range=(1080, 1220),
        active_loans=(1, 3),
        existing_dsr=(0.12, 0.20),
        utilisation=(0.55, 0.75),
        delinquencies=(0, 1),
        max_dpd=(0, 15),
        inquiries=(2, 4),
        income_noise=0.18,
        spend_ratio=(0.80, 0.92),
        negative_days=(3, 10),
        employment_months=(8, 20),
    ),
    "takipte": Persona(
        "takipte",
        "Yasal takipte kredi",
        score_range=(400, 650),
        legal_followup=True,
        delinquencies=(4, 8),
        max_dpd=(90, 180),
        existing_dsr=(0.25, 0.35),
        utilisation=(0.9, 1.0),
        nsf_events=(2, 6),
        negative_days=(40, 90),
    ),
    "kurcalanmis": Persona(
        "kurcalanmis",
        "Kurcalanmış bordro",
        income_truth=0.55,
        spend_ratio=(0.70, 0.85),
    ),
    "serbest": Persona(
        "serbest",
        "Serbest meslek",
        self_employed=True,
        income_noise=0.22,
        score_range=(1300, 1600),
        employment_months=(48, 200),
    ),
}

# Hash-based distribution for identities that are not pinned demo personas.
_DISTRIBUTION: tuple[tuple[str, int], ...] = (
    ("temiz", 50),
    ("gri", 20),
    ("asiri_borclu", 12),
    ("gecikmeli", 8),
    ("ince_dosya", 10),
)

# Pinned demo identities (valid, synthetic TCKNs) for the six demo scenarios.
DEMO_TCKN: dict[str, str] = {
    "temiz": synthetic_tckn("demo-temiz"),
    "ince_dosya": synthetic_tckn("demo-ince-dosya"),
    "yuksek_dsr": synthetic_tckn("demo-yuksek-dsr"),
    "asiri_borclu": synthetic_tckn("demo-asiri-borclu"),
    "kurcalanmis": synthetic_tckn("demo-kurcalanmis"),
    "halka": synthetic_tckn("demo-halka-1"),
    "gri": synthetic_tckn("demo-gri"),
    "gecikmeli": synthetic_tckn("demo-gecikmeli"),
    "takipte": synthetic_tckn("demo-takipte"),
}
_PINNED: dict[str, str] = {tckn: key for key, tckn in DEMO_TCKN.items() if key != "halka"}
_PINNED[DEMO_TCKN["halka"]] = "temiz"

EMPLOYERS = (
    "Yıldız Holding A.Ş.",
    "Anadolu Bilişim Ltd. Şti.",
    "Ege Lojistik San. ve Tic. A.Ş.",
    "Karbon Enerji A.Ş.",
    "Bereket Gıda Pazarlama A.Ş.",
    "Marmara Sağlık Hizmetleri A.Ş.",
)


def digest(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def persona_for(tckn: str) -> Persona:
    if tckn in _PINNED:
        return PERSONAS[_PINNED[tckn]]
    total = sum(weight for _, weight in _DISTRIBUTION)
    pick = digest(f"persona|{tckn}") % total
    for key, weight in _DISTRIBUTION:
        if pick < weight:
            return PERSONAS[key]
        pick -= weight
    return PERSONAS["temiz"]


def _rng(tckn: str, stream: str) -> random.Random:
    return random.Random(digest(f"{stream}|{tckn}"))


def _between(rng: random.Random, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return lo if hi <= lo else rng.uniform(lo, hi)


def _ibetween(rng: random.Random, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return lo if hi <= lo else rng.randint(lo, hi)


def kkb_report(tckn: str, declared_income: float) -> dict:
    """Findeks-style risk report consistent with the persona."""
    persona = persona_for(tckn)
    rng = _rng(tckn, "kkb")
    income = max(declared_income, 1.0)
    loans = _ibetween(rng, persona.active_loans)
    existing_service = round(income * _between(rng, persona.existing_dsr), 2)
    utilisation = round(_between(rng, persona.utilisation), 3)
    card_limit = round(income * rng.uniform(1.0, 3.0), -2) if persona.bureau_hit else 0.0
    outstanding_loans = round(existing_service * rng.uniform(14, 26), 2) if loans else 0.0
    score = _ibetween(rng, persona.score_range) if persona.bureau_hit else None
    delinquencies = _ibetween(rng, persona.delinquencies)
    facilities = []
    for index in range(loans):
        share = existing_service / loans if loans else 0
        facilities.append(
            {
                "type": rng.choice(["İhtiyaç", "Taşıt", "Kredili Mevduat"]),
                "monthly_instalment": round(share, 2),
                "outstanding": round(outstanding_loans / loans, 2),
                "opened": (date(2026, 1, 1) - timedelta(days=90 * (index + 1))).isoformat(),
            }
        )
    risk_class = (
        "KAYIT YOK"
        if score is None
        else "ÇOK DÜŞÜK RİSK"
        if score >= 1500
        else "DÜŞÜK RİSK"
        if score >= 1100
        else "ORTA RİSK"
        if score >= 700
        else "YÜKSEK RİSK"
    )
    total_limit = card_limit + outstanding_loans
    total_debt = card_limit * utilisation + outstanding_loans
    return {
        "provider": "KKB-Findeks (MOCK)",
        "bureau_hit": persona.bureau_hit,
        "score": score,
        "risk_class": risk_class,
        "active_loans": loans,
        "facilities": facilities,
        "monthly_instalments": existing_service,
        "card_limit": card_limit,
        "card_utilisation": utilisation,
        "total_limit": round(total_limit, 2),
        "total_debt": round(total_debt, 2),
        "delinquency_count_24m": delinquencies,
        "max_dpd_24m": _ibetween(rng, persona.max_dpd) if delinquencies else 0,
        "inquiries_6m": _ibetween(rng, persona.inquiries),
        "legal_followup": persona.legal_followup,
        "bireysel_borcluluk_endeksi": round(min(1.0, total_debt / (income * 12)), 3),
    }


def sgk_record(tckn: str, declared_income: float, employment_type: str = "MAASLI") -> dict:
    """e-Devlet / SGK service record (employer, start date, premium days, earnings)."""
    persona = persona_for(tckn)
    rng = _rng(tckn, "sgk")
    months = _ibetween(rng, persona.employment_months)
    start = date(2026, 9, 1) - timedelta(days=int(months * 30.4))
    true_income = declared_income * persona.income_truth
    gross = true_income / 0.71
    earnings = [
        round(gross * (1 + rng.uniform(-persona.income_noise, persona.income_noise) / 2), 2)
        for _ in range(12)
    ]
    self_employed = persona.self_employed or employment_type == "SERBEST"
    return {
        "provider": "e-Devlet SGK (MOCK)",
        "employer": "4B Bağ-Kur (kendi nam ve hesabına)"
        if self_employed
        else EMPLOYERS[digest(tckn) % len(EMPLOYERS)],
        "employment_start": start.isoformat(),
        "employment_months": months,
        "premium_days_12m": 360 if persona.key != "gecikmeli" else rng.randint(240, 330),
        "reported_gross_earnings_12m": earnings,
        "status": "4B" if self_employed else "4A",
        "verified": True,
    }


def gib_record(tckn: str, declared_income: float) -> dict:
    """GİB tax registration (self-employed applicants)."""
    rng = _rng(tckn, "gib")
    return {
        "provider": "GİB Vergi (MOCK)",
        "registered": True,
        "tax_office": rng.choice(["Kadıköy", "Çankaya", "Konak", "Nilüfer"]) + " Vergi Dairesi",
        "activity": rng.choice(["Serbest muhasebeci", "Yazılım danışmanlığı", "Mimarlık"]),
        "declared_annual_income": round(declared_income * 12 * rng.uniform(0.9, 1.1), 2),
        "years_registered": rng.randint(2, 15),
    }


def _salary_day(rng: random.Random) -> int:
    return rng.choice([1, 5, 10, 15, 25])


def open_banking_transactions(
    tckn: str, declared_income: float, months: int = 12, as_of: date | None = None
) -> dict:
    """Twelve months of categorisable account activity (ÖHVPS / GEÇİT style)."""
    persona = persona_for(tckn)
    rng = _rng(tckn, "ob")
    as_of = as_of or date(2026, 9, 1)
    salary = declared_income * persona.income_truth
    rent = round(salary * rng.uniform(0.18, 0.30), -1)
    loan_instalment = round(declared_income * _between(rng, persona.existing_dsr), 2)
    nsf_total = _ibetween(rng, persona.nsf_events)
    neg_target = _ibetween(rng, persona.negative_days)
    running = 0.0
    txs: list[dict] = []
    payday = _salary_day(rng)
    start = date(as_of.year, as_of.month, 1)
    for m in range(months, 0, -1):
        month_start = (start - timedelta(days=31 * m)).replace(day=1)
        paid = salary * (1 + rng.uniform(-persona.income_noise, persona.income_noise))
        if persona.self_employed and rng.random() < 0.15:
            paid *= rng.uniform(0.3, 0.6)
        entries: list[tuple[int, float, str]] = [
            (
                payday,
                round(paid, 2),
                "SERBEST MESLEK TAHSILAT"
                if persona.self_employed
                else "MAAS ODEMESI " + EMPLOYERS[digest(tckn) % len(EMPLOYERS)].split()[0].upper(),
            ),
            (min(payday + 2, 28), -rent, "KIRA ODEMESI FAST"),
            (min(payday + 4, 28), -round(salary * rng.uniform(0.03, 0.05), 2), "ELEKTRIK FATURA"),
            (min(payday + 5, 28), -round(salary * rng.uniform(0.01, 0.02), 2), "TURKCELL FATURA"),
            (min(payday + 6, 28), -round(salary * rng.uniform(0.01, 0.03), 2), "IGDAS DOGALGAZ"),
        ]
        if loan_instalment > 0:
            entries.append((min(payday + 3, 28), -loan_instalment, "KREDI TAKSIT ODEMESI"))
        spend_total = salary * _between(rng, persona.spend_ratio) - rent - loan_instalment
        spend_total = max(spend_total, salary * 0.15)
        card = round(spend_total * rng.uniform(0.35, 0.5), 2)
        entries.append((min(payday + 8, 28), -card, "KREDI KARTI ODEME"))
        remaining = spend_total - card
        gambling = spend_total * persona.gambling_share * 3 if persona.gambling_share else 0.0
        if gambling:
            for _ in range(3):
                entries.append((rng.randint(1, 28), -round(gambling / 3, 2), "NESINE BAHIS ODEME"))
            remaining -= gambling
        for _ in range(rng.randint(6, 10)):
            share = remaining / 8
            kind = rng.choice(
                ["MARKET ALISVERIS", "ATM NAKIT CEKIM", "ONLINE ALISVERIS", "EFT GIDEN"]
            )
            entries.append(
                (rng.randint(1, 28), -round(abs(share) * rng.uniform(0.6, 1.3), 2), kind)
            )
        if rng.random() < 0.25:
            entries.append(
                (rng.randint(1, 28), round(salary * rng.uniform(0.05, 0.2), 2), "FAST GELEN HAVALE")
            )
        if nsf_total and rng.random() < nsf_total / months * 1.5:
            entries.append((rng.randint(1, 28), 0.0, "KARSILIKSIZ CEK IADE"))
        for day, amount, desc in sorted(entries, key=lambda e: e[0]):
            running = round(running + amount, 2)
            txs.append(
                {
                    "date": month_start.replace(day=day).isoformat(),
                    "amount": amount,
                    "description": desc,
                    "_running": running,
                }
            )
    # Choose the opening balance so the running balance dips below zero only for
    # personas with negative-balance history; arithmetic stays exact
    # (opening + sum(amounts) == closing), which the statement tamper check relies on.
    lowest = min((t["_running"] for t in txs), default=0.0)
    cushion = -salary * 0.25 if neg_target else salary * rng.uniform(0.15, 0.9)
    opening = round(cushion - lowest, 2)
    for tx in txs:
        tx["balance_after"] = round(opening + tx.pop("_running"), 2)
    return {
        "provider": "Açık Bankacılık GEÇİT (MOCK)",
        "account": {"currency": "TRY", "opening_balance": opening},
        "transactions": txs,
        "closing_balance": txs[-1]["balance_after"] if txs else opening,
    }
