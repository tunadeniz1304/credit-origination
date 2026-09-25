"""Decision feature snapshot.

The snapshot is the single, flat, JSON-serialisable input of the decision
engine: model features, rule-only signals (KYC, fraud, documents) and the
application parameters. It is persisted with every decision together with its
SHA-256 hash, so any decision can be replayed exactly.

Protected or proxy attributes (gender, age band, province) are **never**
model features; age is used only by the legal eligibility rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

MODEL_FEATURES: tuple[str, ...] = (
    "bureau_score",
    "bureau_hit",
    "delinquency_count_24m",
    "max_dpd_24m",
    "inquiries_6m",
    "active_loans",
    "bureau_utilisation",
    "bureau_behavior_score",
    "dsr",
    "loan_to_income",
    "term_months",
    "log_income",
    "employment_months",
    "income_cv",
    "negative_balance_days",
    "nsf_count",
    "gambling_share",
    "savings_rate",
    "avg_balance_to_income",
)

# +1: higher value -> higher PD; -1: higher value -> lower PD; 0: unconstrained.
MONOTONE: dict[str, int] = {
    "bureau_score": -1,
    "bureau_hit": 0,
    "delinquency_count_24m": 1,
    "max_dpd_24m": 1,
    "inquiries_6m": 1,
    "active_loans": 1,
    "bureau_utilisation": 1,
    "bureau_behavior_score": 1,
    "dsr": 1,
    "loan_to_income": 1,
    "term_months": 1,
    "log_income": -1,
    "employment_months": -1,
    "income_cv": 1,
    "negative_balance_days": 1,
    "nsf_count": 1,
    "gambling_share": 1,
    "savings_rate": -1,
    "avg_balance_to_income": -1,
}

FEATURE_LABELS: dict[str, str] = {
    "bureau_score": "KKB kredi notu",
    "bureau_hit": "KKB kaydı",
    "delinquency_count_24m": "Son 24 ay gecikme sayısı",
    "max_dpd_24m": "Azami gecikme günü",
    "inquiries_6m": "Son 6 ay sorgu sayısı",
    "active_loans": "Aktif kredi sayısı",
    "bureau_utilisation": "Limit kullanım oranı",
    "bureau_behavior_score": "KKB ödeme davranışı skoru (gerçek veriyle eğitilmiş)",
    "dsr": "Borç servis oranı",
    "loan_to_income": "Tutar / yıllık gelir",
    "term_months": "Vade",
    "log_income": "Gelir (log)",
    "employment_months": "Çalışma süresi",
    "income_cv": "Gelir değişkenliği",
    "negative_balance_days": "Eksi bakiye gün sayısı",
    "nsf_count": "Karşılıksız işlem",
    "gambling_share": "Kumar/bahis payı",
    "savings_rate": "Tasarruf oranı",
    "avg_balance_to_income": "Ortalama bakiye / gelir",
}

# Features an applicant can change (for counterfactual explanations).
MUTABLE_FEATURES = ("requested_amount", "term_months", "existing_debt_service")


def annuity_factor(term_months: int, annual_rate: float) -> float:
    """Monthly instalment per 1 TL of principal."""
    i = annual_rate / 12.0
    n = max(1, term_months)
    if i == 0:
        return 1.0 / n
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def model_vector(snapshot: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for name in MODEL_FEATURES:
        raw = snapshot.get(name)
        values.append(float("nan") if raw is None else float(raw))
    return values


def with_loan(
    snapshot: dict[str, Any], *, amount: float, term: int, reference_rate: float
) -> dict[str, Any]:
    """Recompute the loan-dependent features for a different amount/term."""
    updated = dict(snapshot)
    income = max(float(snapshot["monthly_income"]), 1.0)
    instalment = amount * annuity_factor(term, reference_rate)
    updated["requested_amount"] = round(amount, 2)
    updated["term_months"] = term
    updated["new_instalment"] = round(instalment, 2)
    updated["dsr"] = round((float(snapshot["existing_debt_service"]) + instalment) / income, 4)
    updated["loan_to_income"] = round(amount / (income * 12), 4)
    return updated


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_snapshot(
    *,
    product: str,
    requested_amount: float,
    term_months: int,
    declared_income: float,
    verified_income: float | None,
    age: int,
    reference_rate: float,
    bureau: dict[str, Any] | None,
    sgk: dict[str, Any] | None,
    cashflow: dict[str, float] | None,
    kyc: dict[str, Any],
    documents: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the flat snapshot from the stage outputs."""
    bureau = bureau or {}
    cashflow = cashflow or {}
    income = verified_income or declared_income
    bureau_service = float(bureau.get("monthly_instalments") or 0.0)
    cash_service = float(cashflow.get("existing_debt_service") or 0.0)
    existing_service = max(bureau_service, cash_service)
    hit = bool(bureau.get("bureau_hit", False))
    snapshot: dict[str, Any] = {
        "product": product,
        "declared_income": round(declared_income, 2),
        "monthly_income": round(income, 2),
        "log_income": round(math.log(max(income, 1.0)), 4),
        "age": age,
        "existing_debt_service": round(existing_service, 2),
        "existing_dsr": round(existing_service / max(income, 1.0), 4),
        "bureau_hit": 1 if hit else 0,
        "bureau_score": bureau.get("score") if hit else None,
        "delinquency_count_24m": int(bureau.get("delinquency_count_24m") or 0),
        "max_dpd_24m": int(bureau.get("max_dpd_24m") or 0),
        "inquiries_6m": int(bureau.get("inquiries_6m") or 0),
        "active_loans": int(bureau.get("active_loans") or 0),
        "bureau_utilisation": float(bureau.get("card_utilisation") or 0.0),
        "bureau_legal_followup": 1 if bureau.get("legal_followup") else 0,
        "bureau_behavior_score": None,
        "employment_months": int((sgk or {}).get("employment_months") or 0),
        "income_cv": float(cashflow.get("income_cv", 0.35)),
        "negative_balance_days": float(cashflow.get("negative_balance_days", 0.0)),
        "nsf_count": float(cashflow.get("nsf_count", 0.0)),
        "gambling_share": float(cashflow.get("gambling_share", 0.0)),
        "savings_rate": float(cashflow.get("savings_rate", 0.0)),
        "avg_balance_to_income": float(cashflow.get("avg_balance_to_income", 0.0)),
        "cashflow_available": 1 if cashflow else 0,
        "tckn_valid": 1 if kyc.get("tckn_valid", True) else 0,
        "sanctions_match": 1 if kyc.get("sanctions_match") else 0,
        "velocity_30d": int(kyc.get("velocity_30d", 0)),
        "fraud_ring_size": int(kyc.get("fraud_ring_size", 1)),
        "anomaly_score": float(kyc.get("anomaly_score", 0.0)),
        "doc_fraud_score": float(documents.get("max_fraud_score", 0.0)),
        "income_mismatch": 1 if documents.get("income_mismatch") else 0,
        "min_field_confidence": float(documents.get("min_field_confidence", 1.0)),
    }
    if hit:
        from app.decisioning.public_mapping import behaviour_score_for

        snapshot["bureau_behavior_score"] = behaviour_score_for(snapshot)
    return with_loan(
        snapshot, amount=requested_amount, term=term_months, reference_rate=reference_rate
    )
