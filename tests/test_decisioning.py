"""Hybrid decision engine tests: DSL safety, rules, model, limits, replayability."""

from __future__ import annotations

import math

import pytest

from app.cashflow.analysis import analyse
from app.core.rules import load_policy_file, load_pricing, parse_policy, policy_file_text
from app.decisioning.engine import REFERENCE_RATE, decide
from app.decisioning.expressions import UnknownVariableError, UnsafeExpressionError, evaluate
from app.decisioning.features import (
    MODEL_FEATURES,
    MONOTONE,
    annuity_factor,
    build_snapshot,
    snapshot_hash,
    with_loan,
)
from app.decisioning.models import get_models
from app.decisioning.rules_engine import evaluate_rules
from app.integrations import personas


# ------------------------------------------------------------------ DSL
@pytest.mark.parametrize(
    ("expr", "ctx", "expected"),
    [
        ("age < 18", {"age": 17}, True),
        ("age + term / 12 > 75", {"age": 70, "term": 72}, True),
        ("a == 1 and (b > 2 or not c)", {"a": 1, "b": 0, "c": False}, True),
        ("0.2 < x <= 0.5", {"x": 0.5}, True),
        ("max(a, b) - min(a, b) == abs(a - b)", {"a": 3, "b": 7}, True),
        ("score < 600", {"score": None}, False),  # missing data never fires
        ("-x > 0", {"x": -1}, True),
        ("a % 2 == 1", {"a": 3}, True),
    ],
)
def test_expression_evaluation(expr, ctx, expected):
    assert evaluate(expr, ctx) is expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "x.__class__",
        "[1, 2][0]",
        "(lambda: 1)()",
        "open('x')",
        "'abc' == 'abc'",
        "x ** 2",
        "_secret > 1",
        "[a for a in b]",
    ],
)
def test_unsafe_expressions_rejected(expr):
    with pytest.raises(UnsafeExpressionError):
        evaluate(expr, {"x": 1, "b": [1], "_secret": 1})


def test_unknown_variable():
    with pytest.raises(UnknownVariableError):
        evaluate("missing > 1", {})


# ------------------------------------------------------------------ fixtures
def _snapshot(
    persona: str, income: float, amount: float, term: int = 36, *, kyc=None, docs=None, age=35
):
    tckn = personas.DEMO_TCKN[persona]
    ob = personas.open_banking_transactions(tckn, income)
    return build_snapshot(
        product="IHTIYAC",
        requested_amount=amount,
        term_months=term,
        declared_income=income,
        verified_income=None,
        age=age,
        reference_rate=REFERENCE_RATE,
        bureau=personas.kkb_report(tckn, income),
        sgk=personas.sgk_record(tckn, income),
        cashflow=analyse(ob["transactions"], ob["account"]["opening_balance"]).features,
        kyc=kyc or {},
        documents=docs or {},
    )


def _decide(snapshot):
    return decide(
        snapshot, policy=load_policy_file(), models=get_models(), pricing_cfg=load_pricing()
    )


# ------------------------------------------------------------------ rules
def test_policy_file_parses_and_is_versioned():
    policy = parse_policy(policy_file_text())
    assert policy.version == "policy_v2"
    assert {r.action for r in policy.rules} == {"decline", "refer"}
    assert all(r.reason.startswith("R") for r in policy.rules)


def test_knockout_rules_fire():
    snap = _snapshot("temiz", 45_000, 100_000, age=17)
    fired = {r.id for r in evaluate_rules(snap, load_policy_file()) if r.fired}
    assert "KO_YAS_ALT" in fired
    result = _decide(snap)
    assert result.outcome == "OTOMATIK_RET"
    assert result.reason_codes[0].code == "R13_YAS"


def test_legal_followup_declines():
    result = _decide(_snapshot("takipte", 40_000, 100_000, 24))
    assert result.outcome == "OTOMATIK_RET"
    assert "R14_TAKIPTE_KREDI" in result.reason_code_list


# ------------------------------------------------------------------ demo personas
def test_clean_salaried_auto_approved():
    result = _decide(_snapshot("temiz", 45_000, 200_000))
    assert result.outcome == "OTOMATIK_ONAY" and not result.conditional
    assert result.offer_amount == 200_000
    assert result.pricing is not None and result.pricing.annual_rate > 0.3
    assert 300 <= result.score["points"] <= 900


def test_thin_file_approved_thanks_to_cash_flow():
    snap = _snapshot("ince_dosya", 35_000, 150_000)
    assert snap["bureau_hit"] == 0 and snap["bureau_score"] is None
    result = _decide(snap)
    assert result.outcome == "OTOMATIK_ONAY"


def test_high_dsr_gets_conditional_counter_offer_with_counterfactual():
    snap = _snapshot("yuksek_dsr", 45_000, 650_000)
    assert snap["dsr"] > 0.5
    result = _decide(snap)
    assert result.outcome == "OTOMATIK_ONAY" and result.conditional
    assert 0 < result.offer_amount < 650_000
    assert result.limits["dsr_offer"] <= 0.5
    assert result.reason_codes[0].code == "R01_DSR_YUKSEK"
    assert "%" in result.reason_codes[0].text
    assert result.counterfactuals and "TL" in result.counterfactuals[0].text


def test_tampered_documents_refer_to_specialist():
    result = _decide(
        _snapshot(
            "kurcalanmis", 45_000, 250_000, docs={"max_fraud_score": 1.0, "income_mismatch": True}
        )
    )
    assert result.outcome == "UZMAN_INCELEMESI"
    assert {"R11_SAHTECILIK_SUPHESI", "R10_BELGE_UYUMSUZ"} <= set(result.reason_code_list)


def test_fraud_ring_refers():
    result = _decide(_snapshot("halka", 40_000, 180_000, kyc={"fraud_ring_size": 3}))
    assert result.outcome == "UZMAN_INCELEMESI"
    assert "R15_HALKA_SUPHESI" in result.reason_code_list


def test_grey_zone_goes_to_specialist():
    result = _decide(_snapshot("gri", 42_000, 220_000))
    assert result.outcome == "UZMAN_INCELEMESI"
    policy = load_policy_file()
    assert policy.decision.auto_approve_max_pd < result.pd < policy.decision.auto_decline_min_pd


def test_delinquent_declined_with_model_reasons():
    result = _decide(_snapshot("gecikmeli", 40_000, 150_000))
    assert result.outcome == "OTOMATIK_RET"
    assert result.pd >= 0.2
    assert all(r.source == "model" for r in result.reason_codes)
    # Arrears reach the model through the real-data behaviour sub-score (lane B), whose
    # reason (R26) names the payment history; the raw count may still surface as R03.
    assert {"R03_GECIKME_GECMISI", "R26_KKB_DAVRANIS"} & set(result.reason_code_list)
    assert len(result.reason_codes) <= 4


# ------------------------------------------------------------------ determinism / monotonicity
def test_decision_is_deterministic_and_hash_stable():
    snap = _snapshot("gri", 42_000, 220_000)
    first, second = _decide(snap), _decide(dict(snap))
    assert first.model_dump(exclude={"latency_ms"}) == second.model_dump(exclude={"latency_ms"})
    assert first.feature_hash == snapshot_hash(snap)


def test_pd_is_monotone_in_dsr_and_bureau_score():
    models = get_models()
    base = _snapshot("temiz", 45_000, 200_000)
    pds = []
    for amount in (100_000, 300_000, 500_000, 800_000):
        pds.append(
            models.pd_model.predict(
                with_loan(base, amount=amount, term=36, reference_rate=REFERENCE_RATE),
                explain=False,
            ).pd
        )
    assert pds == sorted(pds)
    scores = []
    for score in (900, 1200, 1500, 1800):
        scores.append(models.pd_model.predict({**base, "bureau_score": score}, explain=False).pd)
    assert scores == sorted(scores, reverse=True)


def test_model_metadata_contract():
    meta = get_models().pd_model.meta
    assert meta["features"] == list(MODEL_FEATURES)
    assert meta["monotone_constraints"] == MONOTONE
    for key in ("auc", "gini", "ks", "brier"):
        assert key in meta["metrics"]
    assert meta["metrics"]["auc"] > 0.75
    assert "gender" not in meta["features"] and "age" not in meta["features"]


def test_challenger_is_shadow_only():
    result = _decide(_snapshot("temiz", 45_000, 200_000))
    assert result.challenger["version"].startswith("challenger")
    assert 0 < result.challenger["pd"] < 1


def test_annuity_and_with_loan():
    assert math.isclose(annuity_factor(12, 0.0), 1 / 12)
    snap = with_loan(
        {"monthly_income": 10_000, "existing_debt_service": 1_000},
        amount=100_000,
        term=24,
        reference_rate=0.45,
    )
    assert snap["dsr"] > 0.1 and snap["loan_to_income"] == round(100_000 / 120_000, 4)
