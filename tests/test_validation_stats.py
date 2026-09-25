"""DeLong test, bootstrap CI and calibration diagnostics."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats as sps

from app.validation import stats

# Hand-worked example (3 defaulters, 3 non-defaulters).
Y = np.array([1, 1, 1, 0, 0, 0])
A = np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.2])  # AUC 8/9
B = np.array([0.6, 0.9, 0.3, 0.5, 0.4, 0.1])  # AUC 7/9


# ------------------------------------------------------------------ DeLong
def test_delong_components_match_hand_computation():
    aucs, v10, v01 = stats.delong_components(Y, np.vstack([A, B]))
    assert aucs == pytest.approx([8 / 9, 7 / 9])
    # V10: share of non-defaulters each defaulter out-scores; V01: share of defaulters above.
    assert v10[0] == pytest.approx([1, 1, 2 / 3])
    assert v10[1] == pytest.approx([1, 1, 1 / 3])
    assert v01[0] == pytest.approx([2 / 3, 1, 1])
    assert v01[1] == pytest.approx([2 / 3, 2 / 3, 1])


def test_delong_test_known_example():
    """Var(A)=2/81, Var(B)=5/81, Cov=5/162 ⇒ Var(Δ)=2/81, Δ=1/9 ⇒ z=1/√2."""
    _, cov = stats.delong_covariance(Y, np.vstack([A, B]))
    assert cov[0, 0] == pytest.approx(2 / 81)
    assert cov[1, 1] == pytest.approx(5 / 81)
    assert cov[0, 1] == pytest.approx(5 / 162)
    result = stats.delong_test(Y, A, B)
    assert result["auc_diff"] == pytest.approx(1 / 9, abs=1e-5)
    assert result["z"] == pytest.approx(1 / math.sqrt(2), abs=1e-4)
    assert result["p_value"] == pytest.approx(2 * sps.norm.sf(1 / math.sqrt(2)), rel=1e-4)


def test_delong_identical_scores_and_ties():
    result = stats.delong_test(Y, A, A)
    assert result["z"] == 0.0 and result["p_value"] == 1.0
    tied = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert stats.fast_auc(Y, tied) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        stats.delong_components(np.zeros(4), np.ones(4))


def test_delong_auc_matches_sklearn_on_random_data():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500) + 0.3 * y
    assert stats.fast_auc(y, p) == pytest.approx(roc_auc_score(y, p))
    lo, hi = stats.delong_auc_ci(y, p)
    assert lo < roc_auc_score(y, p) < hi


def test_bootstrap_ci_and_discrimination():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(rng.random(400) * 0.6 + 0.3 * y, 0, 1)
    ci = stats.bootstrap_auc_ci(y, p, iterations=200, confidence=0.95, seed=3)
    metrics = stats.discrimination(y, p, iterations=100, confidence=0.95, seed=3)
    assert ci[0] < metrics["auc"] < ci[1]
    assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1, abs=1e-3)
    assert {"ks", "brier", "log_loss", "auc_ci_delong"} <= set(metrics)


# ------------------------------------------------------------------ calibration
def test_calibration_perfect_and_biased():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.6, 20_000)
    y = (rng.random(20_000) < p).astype(int)
    good = stats.calibration_summary(y, p, bins=10, low_risk_deciles=3)
    assert good["ece"] < 0.02
    assert good["hosmer_lemeshow"]["p_value"] > 0.001
    assert 0.8 < good["low_risk"]["ratio_observed_to_predicted"] < 1.2
    bad = stats.calibration_summary(y, p / 2, bins=10, low_risk_deciles=3)
    assert bad["ece"] > good["ece"]
    assert bad["low_risk"]["ratio_observed_to_predicted"] > 1.5  # under-estimation flagged
    assert bad["hosmer_lemeshow"]["p_value"] < 1e-6
    deciles = bad["deciles"]
    assert [r["decile"] for r in deciles] == list(range(1, 11))
    assert deciles[0]["pd_max"] <= deciles[-1]["pd_min"]


def test_calibration_deciles_keep_tied_pds_together():
    # A calibrated isotonic model has few distinct PDs: a tie block must not be split.
    p = np.r_[np.full(50, 0.01), np.full(300, 0.02), np.linspace(0.03, 0.5, 650)]
    y = (np.random.default_rng(1).random(len(p)) < p).astype(int)
    rows = stats.calibration_deciles(y, p)
    assert sum(r["n"] for r in rows) == len(p)
    tied = [r for r in rows if r["pd_min"] <= 0.02 <= r["pd_max"]]
    assert len(tied) == 1 and tied[0]["n"] >= 300
    for lower, upper in itertools.pairwise(rows):
        assert lower["pd_max"] < upper["pd_min"]
    # Row order does not change the table.
    perm = np.random.default_rng(2).permutation(len(p))
    assert stats.calibration_deciles(y[perm], p[perm]) == rows


def test_ks_is_evaluated_only_between_tie_blocks():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert stats.ks_statistic(y, p) == 0.0
    assert stats.ks_statistic(y[::-1], p) == 0.0
