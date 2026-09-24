"""Cash-flow categorisation and feature tests."""

from __future__ import annotations

import pytest

from app.cashflow.analysis import MODEL_CASHFLOW_FEATURES, analyse, categorise
from app.integrations.personas import DEMO_TCKN, open_banking_transactions


@pytest.mark.parametrize(
    ("description", "amount", "category"),
    [
        ("MAAS ODEMESI YILDIZ", 45000, "gelir_maas"),
        ("KIRA ODEMESI FAST", -12000, "kira"),
        ("TURKCELL FATURA", -500, "fatura"),
        ("KREDI TAKSIT ODEMESI", -3000, "kredi_odeme"),
        ("KREDI KARTI ODEME", -8000, "kart"),
        ("NESINE BAHIS ODEME", -900, "kumar"),
        ("ATM NAKIT CEKIM", -1000, "nakit"),
        ("FAST GELEN HAVALE", 2000, "transfer_gelen"),
        ("EFT GIDEN", -2000, "transfer"),
        ("KARSILIKSIZ CEK IADE", 0, "karsiliksiz"),
        ("BILINMEYEN", -10, "diger"),
    ],
)
def test_categorise(description, amount, category):
    assert categorise(description, amount) == category


def _features(persona: str) -> dict[str, float]:
    ob = open_banking_transactions(DEMO_TCKN[persona], 45_000)
    return analyse(ob["transactions"], ob["account"]["opening_balance"]).features


def test_at_least_twenty_features_and_model_subset():
    features = _features("temiz")
    assert len(features) >= 20
    assert set(MODEL_CASHFLOW_FEATURES) <= set(features)


def test_clean_profile_features():
    f = _features("temiz")
    assert f["income_months"] == 12
    assert f["income_cv"] < 0.05
    assert f["negative_balance_days"] == 0
    assert f["gambling_share"] == 0
    assert f["savings_rate"] > 0.1
    assert abs(f["avg_monthly_income"] - 45_000) / 45_000 < 0.05


def test_delinquent_profile_features():
    f = _features("gecikmeli")
    assert f["negative_balance_days"] > 0
    assert f["nsf_count"] >= 1
    assert f["gambling_share"] > 0
    assert f["savings_rate"] < _features("temiz")["savings_rate"]


def test_tampered_payslip_persona_has_lower_bank_salary():
    f = _features("kurcalanmis")
    assert f["avg_monthly_income"] < 45_000 * 0.7


def test_monthly_series_and_totals():
    ob = open_banking_transactions(DEMO_TCKN["temiz"], 45_000)
    result = analyse(ob["transactions"], ob["account"]["opening_balance"])
    assert len(result.monthly) == 12
    assert result.monthly[0].income > 0
    assert "gelir_maas" in result.category_totals
    assert result.transactions_count == len(ob["transactions"])


def test_empty_transactions():
    result = analyse([], 0.0)
    assert result.features["income_cv"] == 1.0
    assert result.monthly == []
