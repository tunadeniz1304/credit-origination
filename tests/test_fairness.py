"""Fairness at a fixed approval rate and the less-discriminatory-alternative search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.rules import load_validation
from app.governance.fairness import (
    adverse_impact,
    air_spread,
    approve_at_rate,
    group_fairness,
    group_thresholds_at_rate,
    less_discriminatory_search,
    proxy_strength,
    recommend_lda,
)
from app.validation.datasets import FIXTURE_PATH, load_frame, prepare


# ------------------------------------------------------------------ fairness helpers
def test_approve_at_rate_is_exact_and_group_thresholds_equalise():
    scores = np.linspace(0, 1, 100)
    assert approve_at_rate(scores, 0.7).sum() == 70
    groups = pd.Series(["a"] * 50 + ["b"] * 50)
    approved = group_thresholds_at_rate(scores, groups, 0.6)
    assert approved[:50].mean() == approved[50:].mean() == 0.6


def test_group_fairness_rates_and_gaps():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    approved = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    groups = pd.Series(["x"] * 4 + ["y"] * 4)
    out = group_fairness(y, approved, groups)
    assert out["selection_rate"] == {"x": 0.75, "y": 0.25}
    assert out["air"]["y"] == pytest.approx(1 / 3, abs=1e-3)
    assert not out["passes_four_fifths"]
    assert out["tpr_good_approved"] == {"x": 1.0, "y": 0.5}
    assert out["tpr_gap"] == 0.5 and out["fpr_gap"] == 0.5


def test_proxy_strength_detects_a_proxy():
    rng = np.random.default_rng(4)
    groups = pd.Series(rng.choice(["a", "b"], 1000))
    X = pd.DataFrame(
        {"proxy": (groups == "a") + rng.normal(0, 0.3, 1000), "noise": rng.random(1000)}
    )
    strength = proxy_strength(X, groups)
    assert next(iter(strength)) == "proxy" and strength["proxy"] > 0.9
    assert strength["noise"] < 0.6


def _fixture_split():
    cfg = load_validation()
    data = prepare("uci_taiwan", load_frame(FIXTURE_PATH), cfg.fairness)
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(
        np.arange(len(data.y)), test_size=0.25, stratify=data.y, random_state=0
    )
    return data, tr, te


def test_lda_table_rows_share_one_approval_rate():
    data, tr, te = _fixture_split()
    table = less_discriminatory_search(
        data.X.iloc[tr],
        data.y[tr],
        data.X.iloc[te],
        data.y[te],
        data.protected.SEX.iloc[tr],
        data.protected.SEX.iloc[te],
        approval_rate=0.7,
        epsilons=[0.02],
        proxy_auc_threshold=0.99,  # nothing qualifies: the strongest proxy is weakened anyway
        alternatives={"other_family": data.X.iloc[te, 0].rank(pct=True).to_numpy()},
    )
    rates = {r["approval_rate"] for r in table["rows"]}
    assert rates == {0.7}
    kinds = [r["kind"] for r in table["rows"]]
    assert kinds == [
        "baseline",
        "model_family",
        "proxy_removal",
        "exponentiated_gradient",
        "group_threshold",
    ]
    assert table["rows"][1]["family"] == "other_family"
    assert len(table["rows"][2]["dropped_features"]) == 1
    group_row = table["rows"][-1]
    assert group_row["min_air"] >= 0.98 and "legal_caveat" in group_row
    assert table["reference_rows"][0]["comparable"] is False
    rec = recommend_lda(table, max_auc_loss=0.5)
    assert rec["reason"]


def test_recommend_lda_prefers_fairer_row_within_auc_budget():
    table = {
        "rows": [
            {"model": "champ", "kind": "champion", "auc": 0.78, "min_air": 0.70},
            {"model": "weak", "kind": "proxy_removal", "auc": 0.775, "min_air": 0.85},
            {"model": "eg", "kind": "exponentiated_gradient", "auc": 0.70, "min_air": 0.99},
            {"model": "thr", "kind": "group_threshold", "auc": 0.78, "min_air": 1.0},
        ]
    }
    assert recommend_lda(table, 0.01)["recommended"] == "weak"
    assert recommend_lda({"rows": table["rows"][:1]}, 0.01)["recommended"] is None


def test_recommend_lda_considers_other_model_families():
    table = {
        "rows": [
            {"model": "scorecard", "kind": "baseline", "auc": 0.80, "min_air": 0.70},
            {"model": "drop", "kind": "proxy_removal", "auc": 0.70, "min_air": 0.95},
            {"model": "lightgbm", "kind": "model_family", "auc": 0.795, "min_air": 0.78},
            {"model": "thr", "kind": "group_threshold", "auc": 0.80, "min_air": 1.0},
        ]
    }
    rec = recommend_lda(table, 0.01)
    assert rec["recommended"] == "lightgbm" and rec["kind"] == "model_family"
    assert rec["min_air_to"] == 0.78


def test_recommend_lda_reports_untestable_attribute():
    table = {"rows": [{"model": "champ", "kind": "baseline", "auc": 0.8, "min_air": None}]}
    rec = recommend_lda(table, 0.01)
    assert rec["recommended"] is None and rec["testable"] is False


def test_tie_break_is_seeded_and_independent_of_row_order():
    scores = np.array([0.1] * 4 + [0.5] * 6)  # cut-off at 7 falls inside the 0.5 tie block
    first = approve_at_rate(scores, 0.7, seed=3)
    assert first.sum() == 7 and first[:4].all()
    assert (approve_at_rate(scores, 0.7, seed=3) == first).all()
    # Row order no longer decides: over seeds every tied applicant is sometimes declined.
    declined = sum(1 - approve_at_rate(scores, 0.7, seed=s)[4:] for s in range(40))
    assert (declined > 0).all()
    # Rows 4.. used to be approved first-come; a stable sort would always decline the last 3.
    assert not all((approve_at_rate(scores, 0.7, seed=s)[-3:] == 0).all() for s in range(40))


def test_single_testable_group_is_reported_as_not_testable():
    groups = pd.Series(["a"] * 95 + ["b"] * 5)
    approved = np.r_[np.ones(60, dtype=int), np.zeros(40, dtype=int)]
    stats = adverse_impact(approved, groups, min_group_size=20)
    assert stats["testable"] is False
    assert stats["min_air"] is None and stats["passes_four_fifths"] is None
    assert stats["air"] == {"a": None}
    fairness = group_fairness(np.zeros(100), approved, groups, min_group_size=20)
    assert fairness["passes_four_fifths"] is None and fairness["tpr_gap"] is None


def test_air_spread_over_tie_break_seeds():
    rng = np.random.default_rng(5)
    groups = pd.Series(rng.choice(["a", "b"], 400))
    scores = np.round(rng.random(400), 1)  # heavy ties
    y = (rng.random(400) < scores).astype(int)
    spread = air_spread(y, scores, groups, approval_rate=0.5, seeds=list(range(10)))
    assert spread is not None and spread["seeds"] == 10
    assert spread["min"] <= spread["median"] <= spread["max"]
    assert 0 <= spread["share_passing"] <= 1
    single = pd.Series(["a"] * 400)
    assert air_spread(y, scores, single, approval_rate=0.5, seeds=[0, 1]) is None
