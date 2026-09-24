"""Open-banking cash-flow analytics (Plaid LendScore style).

Transactions are categorised with Turkish keyword rules (salary, rent, bills,
loan instalments, card payments, gambling/betting, cash, transfers, returned
items) and summarised into 20+ features plus monthly chart series. The
feature names used by the PD model are listed in
:data:`MODEL_CASHFLOW_FEATURES`.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

CATEGORY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gelir_maas", re.compile(r"MAAS|MAAŞ|UCRET ODEME|SERBEST MESLEK TAHSILAT", re.I)),
    ("karsiliksiz", re.compile(r"KARSILIKSIZ|KARŞILIKSIZ|IADE", re.I)),
    ("kira", re.compile(r"KIRA", re.I)),
    (
        "fatura",
        re.compile(r"FATURA|ELEKTRIK|DOGALGAZ|DOĞALGAZ|IGDAS|TURKCELL|VODAFONE|TELEKOM|SU ", re.I),
    ),
    ("kredi_odeme", re.compile(r"KREDI TAKSIT|KREDİ TAKSİT|KREDI ODEME", re.I)),
    ("kart", re.compile(r"KREDI KARTI|KART ODEME", re.I)),
    ("kumar", re.compile(r"BAHIS|BAHİS|IDDAA|İDDAA|CASINO|BET|MISLI|NESINE|TUTTUR", re.I)),
    ("nakit", re.compile(r"ATM|NAKIT", re.I)),
    ("transfer", re.compile(r"FAST|EFT|HAVALE", re.I)),
    ("alisveris", re.compile(r"MARKET|ALISVERIS|ALIŞVERİŞ", re.I)),
)

MODEL_CASHFLOW_FEATURES = (
    "income_cv",
    "negative_balance_days",
    "nsf_count",
    "gambling_share",
    "savings_rate",
    "avg_balance_to_income",
)


class Transaction(BaseModel):
    date: date
    amount: float
    description: str
    balance_after: float
    category: str = "diger"


class MonthlyPoint(BaseModel):
    month: str
    income: float
    spending: float
    net: float
    end_balance: float
    gambling: float


class CashflowResult(BaseModel):
    features: dict[str, float]
    monthly: list[MonthlyPoint]
    category_totals: dict[str, float]
    transactions_count: int
    categorised: list[Transaction] = Field(default_factory=list, exclude=True)


def categorise(description: str, amount: float) -> str:
    for category, pattern in CATEGORY_RULES:
        if pattern.search(description):
            if category == "transfer" and amount > 0:
                return "transfer_gelen"
            return category
    return "diger_gelir" if amount > 0 else "diger"


def _daily_balances(txs: list[Transaction], opening: float) -> list[float]:
    """End-of-day balance for every calendar day in the observed window."""
    if not txs:
        return []
    by_day: dict[date, float] = {}
    for tx in txs:
        by_day[tx.date] = tx.balance_after
    days: list[float] = []
    current = opening
    cursor = txs[0].date
    last = txs[-1].date
    while cursor <= last:
        current = by_day.get(cursor, current)
        days.append(current)
        cursor = date.fromordinal(cursor.toordinal() + 1)
    return days


def analyse(raw_transactions: Iterable[dict[str, Any]], opening_balance: float) -> CashflowResult:
    txs = sorted(
        (
            Transaction(
                date=date.fromisoformat(t["date"]),
                amount=float(t["amount"]),
                description=t["description"],
                balance_after=float(t["balance_after"]),
                category=categorise(t["description"], float(t["amount"])),
            )
            for t in raw_transactions
        ),
        key=lambda t: t.date,
    )
    monthly_income: dict[str, float] = defaultdict(float)
    monthly_spend: dict[str, float] = defaultdict(float)
    monthly_gamble: dict[str, float] = defaultdict(float)
    monthly_end: dict[str, float] = {}
    monthly_loan: dict[str, float] = defaultdict(float)
    totals: dict[str, float] = defaultdict(float)
    salary_days: list[int] = []
    for tx in txs:
        key = tx.date.strftime("%Y-%m")
        totals[tx.category] += tx.amount
        monthly_end[key] = tx.balance_after
        if tx.category == "gelir_maas":
            monthly_income[key] += tx.amount
            salary_days.append(tx.date.day)
        elif tx.amount < 0:
            monthly_spend[key] += -tx.amount
            if tx.category == "kumar":
                monthly_gamble[key] += -tx.amount
            if tx.category == "kredi_odeme":
                monthly_loan[key] += -tx.amount
    months = sorted(set(monthly_end))
    incomes = [monthly_income.get(m, 0.0) for m in months]
    spends = [monthly_spend.get(m, 0.0) for m in months]
    avg_income = statistics.fmean(incomes) if incomes else 0.0
    income_cv = (statistics.pstdev(incomes) / avg_income) if avg_income > 0 else 1.0
    total_spend = sum(spends)
    total_income = sum(incomes)
    inflows = sum(t.amount for t in txs if t.amount > 0)
    daily = _daily_balances(txs, opening_balance)
    gambling_total = sum(monthly_gamble.values())
    outflows = [-t.amount for t in txs if t.amount < 0]
    half = max(1, len(incomes) // 2)
    drawdowns: list[float] = []
    for index, tx in enumerate(txs):
        if tx.category == "gelir_maas" and tx.amount > 0:
            window = [
                t.balance_after for t in txs[index : index + 12] if (t.date - tx.date).days <= 5
            ]
            if window:
                drawdowns.append((tx.balance_after - min(window)) / tx.amount)

    features: dict[str, float] = {
        "avg_monthly_income": round(avg_income, 2),
        "income_cv": round(income_cv, 4),
        "income_months": float(sum(1 for v in incomes if v > 0)),
        "months_observed": float(len(months)),
        "min_balance": round(min(daily), 2) if daily else 0.0,
        "avg_balance": round(statistics.fmean(daily), 2) if daily else 0.0,
        "negative_balance_days": float(sum(1 for b in daily if b < 0)),
        "nsf_count": float(sum(1 for t in txs if t.category == "karsiliksiz")),
        "existing_debt_service": round(statistics.fmean(monthly_loan.values()), 2)
        if monthly_loan
        else 0.0,
        "card_payments_avg": round(-totals.get("kart", 0.0) / max(len(months), 1), 2),
        "rent_avg": round(-totals.get("kira", 0.0) / max(len(months), 1), 2),
        "bills_avg": round(-totals.get("fatura", 0.0) / max(len(months), 1), 2),
        "spend_to_income": round(total_spend / total_income, 4) if total_income else 2.0,
        "gambling_share": round(gambling_total / total_spend, 4) if total_spend else 0.0,
        "gambling_count": float(sum(1 for t in txs if t.category == "kumar")),
        "savings_rate": round((total_income - total_spend) / total_income, 4)
        if total_income
        else -1.0,
        "cash_withdrawal_share": round(-totals.get("nakit", 0.0) / total_spend, 4)
        if total_spend
        else 0.0,
        "transfer_in_share": round(totals.get("transfer_gelen", 0.0) / inflows, 4)
        if inflows
        else 0.0,
        "avg_balance_to_income": round(statistics.fmean(daily) / avg_income, 4)
        if daily and avg_income
        else 0.0,
        "largest_outflow_ratio": round(max(outflows) / avg_income, 4)
        if outflows and avg_income
        else 0.0,
        "salary_day_std": round(statistics.pstdev(salary_days), 3) if len(salary_days) > 1 else 0.0,
        "end_of_month_balance_avg": round(statistics.fmean(monthly_end.values()), 2)
        if monthly_end
        else 0.0,
        "income_trend": round(
            (statistics.fmean(incomes[half:]) - statistics.fmean(incomes[:half])) / avg_income, 4
        )
        if avg_income and len(incomes) >= 2
        else 0.0,
        "post_payday_drawdown": round(statistics.fmean(drawdowns), 4) if drawdowns else 0.0,
    }
    monthly = [
        MonthlyPoint(
            month=m,
            income=round(monthly_income.get(m, 0.0), 2),
            spending=round(monthly_spend.get(m, 0.0), 2),
            net=round(monthly_income.get(m, 0.0) - monthly_spend.get(m, 0.0), 2),
            end_balance=round(monthly_end.get(m, 0.0), 2),
            gambling=round(monthly_gamble.get(m, 0.0), 2),
        )
        for m in months
    ]
    return CashflowResult(
        features=features,
        monthly=monthly,
        category_totals={k: round(v, 2) for k, v in sorted(totals.items())},
        transactions_count=len(txs),
        categorised=txs,
    )


def salary_deposits(result: CashflowResult) -> float:
    """Average monthly salary observed on the account (for reconciliation)."""
    return result.features.get("avg_monthly_income", 0.0)
