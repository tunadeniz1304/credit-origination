"""Fairness at a fixed approval rate and the less-discriminatory-alternative search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.rules import load_validation
from app.governance.fairness import (
    adverse_impact,
    air_spread,
    air_verdict,
    approve_at_rate,
    bootstrap_air_gain,
    bootstrap_min_air,
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


def test_recommend_lda_requires_out_of_fold_loss_within_limit():
    """A fairer row within the limit on a small hold-out but not out-of-fold is rejected."""
    table = {
        "rows": [
            {
                "model": "scorecard",
                "kind": "champion",
                "auc": 0.779,
                "oof_auc": 0.796,
                "min_air": 0.70,
            },
            {
                "model": "lightgbm",
                "kind": "model_family",
                "auc": 0.770,
                "oof_auc": 0.776,
                "min_air": 0.78,
            },
            {
                "model": "thr",
                "kind": "group_threshold",
                "auc": 0.779,
                "oof_auc": 0.796,
                "min_air": 1.0,
            },
        ]
    }
    rec = recommend_lda(table, 0.01)
    assert rec["recommended"] is None and rec["basis"] == "out_of_fold_and_holdout"
    assert rec["rejected"] == [
        {
            "model": "lightgbm",
            "kind": "model_family",
            "min_air": 0.78,
            "auc_loss": 0.009,
            "oof_auc_loss": 0.02,
        }
    ]
    # Within the limit on both bases: recommended, with both losses reported.
    table["rows"][1]["oof_auc"] = 0.790
    rec = recommend_lda(table, 0.01)
    assert rec["recommended"] == "lightgbm" and rec["rejected"] == []
    assert (rec["auc_loss"], rec["oof_auc_loss"]) == (0.009, 0.006)
    # Out-of-fold within, hold-out outside the limit: rejected as well.
    table["rows"][1]["auc"] = 0.760
    assert recommend_lda(table, 0.01)["recommended"] is None
    # A row without an out-of-fold AUC cannot pass when the anchor has one.
    del table["rows"][1]["oof_auc"]
    table["rows"][1]["auc"] = 0.779
    assert recommend_lda(table, 0.01)["recommended"] is None


def test_lda_search_reports_out_of_fold_auc_for_every_row():
    data, tr, te = _fixture_split()
    rng = np.random.default_rng(0)
    table = less_discriminatory_search(
        data.X.iloc[tr],
        data.y[tr],
        data.X.iloc[te],
        data.y[te],
        data.protected.SEX.iloc[tr],
        data.protected.SEX.iloc[te],
        approval_rate=0.7,
        epsilons=[0.05],
        proxy_auc_threshold=0.99,
        baseline_scores=rng.random(len(te)),
        alternatives={"other": rng.random(len(te))},
        cv_folds=3,
        baseline_oof=rng.random(len(tr)),
        alternatives_oof={"other": rng.random(len(tr))},
    )
    for row in table["rows"]:
        assert 0.3 < row["oof_auc"] < 1, row["model"]
    fitted = [r for r in table["rows"] if r["kind"] in ("baseline", "proxy_removal")]
    assert all(r["oof_auc"] > 0.65 for r in fitted)  # genuinely refitted per fold
    assert recommend_lda(table, 0.01)["basis"] == "out_of_fold_and_holdout"


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


# ------------------------------------------------------------------ sampling error of the AIR
def _age_like_holdout() -> tuple[np.ndarray, pd.Series]:
    """German-like hold-out: groups 78/66/34/22, the 22-row reference approves 19/22."""
    sizes = {"21-29": (78, 47), "30-39": (66, 48), "40-49": (34, 26), "50+": (22, 19)}
    labels, approved = [], []
    for group, (n, k) in sizes.items():
        labels += [group] * n
        approved += [1] * k + [0] * (n - k)
    return np.array(approved), pd.Series(labels)


def test_bootstrap_min_air_is_wide_for_small_groups_and_seeded():
    approved, groups = _age_like_holdout()
    point = adverse_impact(approved, groups, min_group_size=20)["min_air"]
    assert point < 0.8  # the point estimate alone "fails" the four-fifths rule
    boot = bootstrap_min_air(approved, groups, min_group_size=20, iterations=1000, seed=7)
    lo, hi = boot["ci"]
    assert lo < point < hi and lo < 0.8 < hi
    assert boot["verdict"] == "inconclusive" and 0 < boot["share_passing"] < 0.5
    assert boot["method"] == "stratified_bootstrap_fixed_decisions"
    assert bootstrap_min_air(approved, groups, min_group_size=20, iterations=1000, seed=7) == boot
    # Not testable: fewer than two groups above the size floor.
    assert bootstrap_min_air(approved, groups, min_group_size=70, iterations=50) is None


def test_bootstrap_min_air_establishes_clear_results_on_large_groups():
    groups = pd.Series(["a"] * 4000 + ["b"] * 4000)
    unequal = np.r_[np.ones(3200), np.zeros(800), np.ones(1600), np.zeros(2400)]
    assert bootstrap_min_air(unequal, groups, iterations=300)["verdict"] == "fails"
    rng = np.random.default_rng(1)
    equal = (rng.random(8000) < 0.7).astype(int)
    assert bootstrap_min_air(equal, groups, iterations=300)["verdict"] == "passes"
    assert air_verdict([0.55, 0.89]) == "inconclusive"
    assert air_verdict([0.81, 0.95]) == "passes" and air_verdict([0.5, 0.79]) == "fails"


def test_bootstrap_air_gain_is_paired():
    approved, groups = _age_like_holdout()
    same = bootstrap_air_gain(approved, approved, groups, min_group_size=20, iterations=200)
    assert same["ci"] == [0.0, 0.0] and same["established"] is False
    # With a 22-row reference group even a large point gain is not established.
    labels = groups.to_numpy()
    fairer = approved.copy()
    for group, k in {"21-29": 66, "30-39": 56, "40-49": 29}.items():  # all groups near 85%
        idx = np.flatnonzero(labels == group)
        fairer[idx] = 0
        fairer[idx[:k]] = 1
    small = bootstrap_air_gain(fairer, approved, groups, min_group_size=20, iterations=500)
    assert small["ci"][0] <= 0 < small["ci"][1] and small["established"] is False
    big = pd.Series(["a"] * 400 + ["b"] * 400)
    anchor = np.r_[np.ones(320), np.zeros(80), np.ones(200), np.zeros(200)]  # AIR 0.625
    even = np.r_[np.ones(320), np.zeros(80), np.ones(300), np.zeros(100)]  # AIR 0.9375
    gain = bootstrap_air_gain(even, anchor, big, iterations=500)
    assert gain["ci"][0] > 0 and gain["established"] is True


def test_lda_rows_carry_bootstrap_intervals_and_recommendation_reports_the_gain():
    rng = np.random.default_rng(3)
    n = 1200
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    group = pd.Series(np.where(X.x2 + rng.normal(scale=0.5, size=n) > 0, "a", "b"))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X.x1 + X.x2)))).astype(int)
    table = less_discriminatory_search(
        X.iloc[:800],
        y[:800],
        X.iloc[800:],
        y[800:],
        group.iloc[:800],
        group.iloc[800:],
        approval_rate=0.7,
        epsilons=[0.05],
        proxy_auc_threshold=0.6,
        bootstrap_iterations=100,
    )
    rows = table["rows"]
    assert all("_approved" not in r for r in rows + table["reference_rows"])
    assert all(r["min_air_ci"][0] <= r["min_air_ci"][1] for r in rows)
    assert "min_air_gain_ci" not in rows[0] and all("min_air_gain_ci" in r for r in rows[1:])
    rec = recommend_lda(table, max_auc_loss=1.0)
    assert rec["recommended"] is not None
    assert rec["air_gain_established"] == (rec["air_gain_ci"][0] > 0)
    assert "GA" in rec["reason"]
