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


# ------------------------------------------------------------------ materiality (audit F02)
_CTX = {
    "dsr": 0.30,
    "max_dsr": 0.5,
    "bureau_hit": 1,
    "bureau_score": 1500,
    "employment_months": 52,
    "inquiries_6m": 6,
    "loan_to_income": 0.9,
    "savings_rate": 0.02,
}


def test_tenure_in_healthy_range_never_reads_as_short():
    """Audit case: 52 months of tenure on a PD≈0.35% file produced R05 'short tenure'."""
    reasons = build_reason_codes(
        context=_CTX, fired_rules=[], shap_values={"employment_months": 0.8}, points_lost={}
    )
    assert reasons == []
    short = build_reason_codes(
        context={**_CTX, "employment_months": 8},
        fired_rules=[],
        shap_values={"employment_months": 0.8},
    )
    assert [r.code for r in short] == ["R05_ISTIHDAM_KISA"]


def test_immaterial_contributions_are_dropped():
    reasons = build_reason_codes(
        context=_CTX,
        fired_rules=[],
        shap_values={"inquiries_6m": 0.02, "loan_to_income": 0.5},
        points_lost={"loan_to_income": 1.0},  # below the points-lost threshold
    )
    assert reasons == []
    material = build_reason_codes(
        context=_CTX,
        fired_rules=[],
        shap_values={"inquiries_6m": 0.3, "loan_to_income": 0.5},
        points_lost={"loan_to_income": 12.0},
    )
    assert [r.code for r in material] == ["R09_TUTAR_GELIR", "R04_SORGU_SAYISI"]


def test_at_most_four_codes_rules_first():
    reasons = build_reason_codes(
        context={**_CTX, "employment_months": 6, "gambling_share": 0.2, "nsf_count": 3},
        fired_rules=[{"reason": "R12_YAPTIRIM_ESLESME"}, {"reason": "R16_VELOCITY"}],
        shap_values={
            "inquiries_6m": 0.9,
            "employment_months": 0.8,
            "gambling_share": 0.7,
            "nsf_count": 0.6,
        },
    )
    assert [r.code for r in reasons] == [
        "R12_YAPTIRIM_ESLESME",
        "R16_VELOCITY",
        "R04_SORGU_SAYISI",
        "R05_ISTIHDAM_KISA",
    ]


def test_approved_files_get_improvement_language():
    reasons = build_reason_codes(
        context=_CTX,
        fired_rules=[],
        shap_values={"loan_to_income": 0.5},
        approved=True,
    )
    assert reasons[0].kind == "improvement"
    assert reasons[0].text.startswith("İyileştirme alanı")
    assert "yüksek" not in reasons[0].text


def test_high_dsr_inside_cap_is_not_called_above_the_limit():
    near = build_reason_codes(
        context={**_CTX, "dsr": 0.47}, fired_rules=[], shap_values={"dsr": 0.6}
    )
    assert [r.code for r in near] == ["R27_DSR_SINIRDA"]
    assert "yakın" in near[0].text and "üzerinde" not in near[0].text
    above = build_reason_codes(
        context={**_CTX, "dsr": 0.58}, fired_rules=[], shap_values={"dsr": 0.6}
    )
    assert [r.code for r in above] == ["R01_DSR_YUKSEK"]
    low = build_reason_codes(context=_CTX, fired_rules=[], shap_values={"dsr": 0.6})
    assert low == []  # 30 % is not in the adverse range (> 80 % of the cap)


def test_unknown_variable_in_condition_never_invents_a_reason():
    from app.core.rules import load_reasons
    from app.explainability.reasons import is_adverse

    catalog = load_reasons()
    assert is_adverse("employment_months", {}, catalog) is False
    assert is_adverse("not_configured", {}, catalog) is True
