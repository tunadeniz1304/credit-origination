"""Fairness at a fixed approval rate and the less-discriminatory-alternative search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.rules import load_validation
from app.governance.fairness import (
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
    )
    rates = {r["approval_rate"] for r in table["rows"]}
    assert rates == {0.7}
    kinds = [r["kind"] for r in table["rows"]]
    assert kinds == ["baseline", "proxy_removal", "exponentiated_gradient", "group_threshold"]
    assert len(table["rows"][1]["dropped_features"]) == 1
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
