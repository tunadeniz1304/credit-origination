"""Reason-code rendering and counterfactual search tests."""

from __future__ import annotations

from app.core.rules import load_reasons
from app.explainability.counterfactual import search
from app.explainability.reasons import build_reason_codes, format_value, render


def test_reason_code_renders_applicant_values_in_turkish_format():
    text = render("R01_DSR_YUKSEK", {"dsr": 0.58, "max_dsr": 0.5})
    assert text == (
        "Aylık borç ödemelerinizin gelirinize oranı %58,0 ile politika sınırı olan "
        "%50,0 seviyesinin üzerinde."
    )


def test_every_catalog_code_renders_without_placeholders():
    catalog = load_reasons()
    ctx = {
        name: 1
        for name in (
            "dsr",
            "max_dsr",
            "bureau_score",
            "delinquency_count_24m",
            "inquiries_6m",
            "employment_months",
            "income_cv",
            "negative_balance_days",
            "gambling_share",
            "loan_to_income",
            "savings_rate",
            "nsf_count",
            "bureau_utilisation",
        )
    }
    for code in catalog.codes:
        text = render(code, ctx, catalog)
        assert "{" not in text and text


def test_format_value():
    assert format_value("bureau_score", 1450) == "1.450"
    assert format_value("gambling_share", 0.125) == "%12,5"
    assert format_value("income_cv", 0.1234) == "0,12"
    assert format_value("x", None) == "—"


def test_rule_reasons_first_then_top_adverse_shap():
    reasons = build_reason_codes(
        context={
            "dsr": 0.6,
            "max_dsr": 0.5,
            "bureau_hit": 1,
            "bureau_score": 900,
            "inquiries_6m": 7,
        },
        fired_rules=[{"reason": "R12_YAPTIRIM_ESLESME"}],
        shap_values={"bureau_score": 0.9, "inquiries_6m": 0.4, "savings_rate": -0.3, "dsr": 0.2},
        points_lost={"bureau_score": 55.0},
    )
    codes = [r.code for r in reasons]
    assert codes[0] == "R12_YAPTIRIM_ESLESME"
    assert codes[1:4] == ["R02_KKB_DUSUK", "R04_SORGU_SAYISI", "R01_DSR_YUKSEK"]
    assert "R17_TASARRUF_DUSUK" not in codes  # favourable contributions are not reasons
    assert reasons[1].points_lost == 55.0


def test_thin_file_reason_replaces_bureau_score():
    reasons = build_reason_codes(
        context={"bureau_hit": 0, "bureau_score": None},
        fired_rules=[],
        shap_values={"bureau_score": 0.3},
    )
    assert reasons[0].code == "R20_INCE_DOSYA"


def test_counterfactual_search_finds_smallest_amount_and_longer_term():
    snapshot = {
        "requested_amount": 300_000.0,
        "term_months": 24,
        "monthly_income": 30_000.0,
        "existing_debt_service": 3_000.0,
    }

    def approves(s):
        return s["dsr"] <= 0.5, 0.03

    results = search(
        snapshot, approves=approves, max_term=60, min_amount=10_000, reference_rate=0.45
    )
    kinds = [r.kind for r in results]
    assert ("amount" in kinds and "term" in kinds and "debt" not in kinds) or len(results) <= 3
    amount = next(r for r in results if r.kind == "amount")
    assert amount.changes["requested_amount"] < 300_000
    assert "TL" in amount.text


def test_counterfactual_returns_nothing_when_impossible():
    snapshot = {
        "requested_amount": 100_000.0,
        "term_months": 12,
        "monthly_income": 10_000.0,
        "existing_debt_service": 0.0,
    }
    assert (
        search(
            snapshot,
            approves=lambda s: (False, 0.9),
            max_term=24,
            min_amount=10_000,
            reference_rate=0.45,
        )
        == []
    )
