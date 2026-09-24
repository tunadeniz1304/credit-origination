"""Lane A: the production modelling recipe run on real public data.

The same helpers that train the production models (:mod:`app.decisioning.training`)
fit a monotone LightGBM with isotonic calibration, a logistic regression and
an optbinning WoE scorecard on each public set's own variables.

Design (no time axis in the public sets):

* stratified hold-out (``holdout_fraction``) that no model sees during fitting;
* stratified ``cv_folds``-fold cross-validation on the remainder → per-fold
  AUC and out-of-fold scores (the isotonic map of LightGBM is fitted on the
  out-of-fold scores, never on the hold-out);
* hold-out metrics with bootstrap and DeLong confidence intervals, pairwise
  DeLong tests, decile calibration (low-risk deciles separately), ECE and
  Hosmer–Lemeshow;
* evidence-based champion choice (:func:`select_champion`);
* fairness at the same approval rate and the LDA trade-off table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.core.rules import ValidationConfig
from app.decisioning.training import fit_isotonic, fit_lightgbm, make_logistic, make_scorecard
from app.governance.fairness import (
    approve_at_rate,
    group_fairness,
    less_discriminatory_search,
    recommend_lda,
)
from app.validation.datasets import PreparedData
from app.validation.stats import calibration_summary, delong_test, discrimination

MODEL_LABELS = {
    "lightgbm": "Monotonik LightGBM + izotonik kalibrasyon",
    "logistic": "Lojistik regresyon",
    "scorecard": "WoE skor kartı (optbinning + lojistik)",
}
EARLY_STOP_FRACTION = 0.1


@dataclass
class LaneAResult:
    metrics: dict[str, Any]
    y_holdout: np.ndarray
    predictions: dict[str, np.ndarray]


class _LightGBM:
    def __init__(self, monotone: dict[str, int], seed: int) -> None:
        self.monotone = monotone
        self.seed = seed
        self.booster: Any = None
        self.iso: Any = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _LightGBM:
        from sklearn.model_selection import train_test_split

        xtr, xva, ytr, yva = train_test_split(
            X, y, test_size=EARLY_STOP_FRACTION, stratify=y, random_state=self.seed
        )
        self.booster = fit_lightgbm(xtr, ytr, xva, yva, self.monotone)
        return self

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(X, num_iteration=self.booster.best_iteration))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw(X)
        return np.asarray(self.iso.predict(raw)) if self.iso is not None else raw


class _Sklearn:
    def __init__(self, model: Any) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _Sklearn:
        self.model.fit(X, y)
        return self

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X)[:, 1])

    predict = raw


def _builders(data: PreparedData, seed: int) -> dict[str, Any]:
    features = list(data.X.columns)
    return {
        "lightgbm": lambda: _LightGBM(data.monotone, seed),
        "logistic": lambda: _Sklearn(make_logistic()),
        "scorecard": lambda: _Sklearn(make_scorecard(features, data.monotone)),
    }


VERDICTS = {
    "selected": "seçildi",
    "not_significant": "anlamlı değil",
    "immaterial": "pratikte önemsiz",
    "worse": "anlamlı biçimde daha zayıf",
}


def select_champion(
    holdout: dict[str, dict[str, Any]],
    tests: dict[str, dict[str, Any]],
    *,
    alpha: float,
    min_gain: float,
    simplicity_order: list[str],
) -> dict[str, Any]:
    """Prefer the simplest model unless a more complex one is significantly *and* materially better."""
    order = [m for m in simplicity_order if m in holdout]
    champion = order[0]
    comparisons: list[dict[str, Any]] = []
    for challenger in order[1:]:
        test = tests.get(f"{challenger}_vs_{champion}")
        if test is None:
            flipped = tests[f"{champion}_vs_{challenger}"]
            test = {**flipped, "auc_diff": -flipped["auc_diff"]}
        gain, p = float(test["auc_diff"]), float(test["p_value"])
        significant = p < alpha
        if significant and gain >= min_gain:
            verdict = "selected"
        elif not significant:
            verdict = "not_significant"
        elif gain < 0:
            verdict = "worse"
        else:
            verdict = "immaterial"
        comparisons.append(
            {
                "challenger": challenger,
                "incumbent": champion,
                "auc_diff": round(gain, 5),
                "p_value": p,
                "verdict": verdict,
            }
        )
        if verdict == "selected":
            champion = challenger
    return {
        "model": champion,
        "label": MODEL_LABELS[champion],
        "alpha": alpha,
        "min_auc_gain": min_gain,
        "simplicity_order": simplicity_order,
        "comparisons": comparisons,
        "steps": [
            f"{MODEL_LABELS[c['challenger']]} vs {MODEL_LABELS[c['incumbent']]}: "
            f"ΔAUC={c['auc_diff']:+.4f}, DeLong p={c['p_value']:.3g} → {VERDICTS[c['verdict']]}"
            for c in comparisons
        ],
    }


def _shap_top(model: _LightGBM, X: pd.DataFrame, seed: int, rows: int = 2000) -> list[dict]:
    import shap

    sample = X.sample(min(rows, len(X)), random_state=seed)
    values = shap.TreeExplainer(model.booster).shap_values(sample)
    matrix = values[1] if isinstance(values, list) else values
    mean_abs = np.abs(np.asarray(matrix)).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    return [
        {"feature": str(sample.columns[i]), "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order
    ]


def run_lane_a(data: PreparedData, cfg: ValidationConfig) -> LaneAResult:
    from sklearn.model_selection import StratifiedKFold, train_test_split

    seed = cfg.seed
    idx_train, idx_hold = train_test_split(
        np.arange(len(data.y)), test_size=cfg.holdout_fraction, stratify=data.y, random_state=seed
    )
    X_train, X_hold = data.X.iloc[idx_train], data.X.iloc[idx_hold]
    y_train, y_hold = data.y[idx_train], data.y[idx_hold]
    builders = _builders(data, seed)

    from sklearn.metrics import roc_auc_score

    cv: dict[str, dict[str, Any]] = {}
    oof: dict[str, np.ndarray] = {}
    folds = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
    for name, build in builders.items():
        scores = np.zeros(len(y_train))
        aucs = []
        for fit_idx, val_idx in folds.split(X_train, y_train):
            model = build().fit(X_train.iloc[fit_idx], y_train[fit_idx])
            scores[val_idx] = model.raw(X_train.iloc[val_idx])
            aucs.append(float(roc_auc_score(y_train[val_idx], scores[val_idx])))
        oof[name] = scores
        cv[name] = {
            "fold_auc": [round(a, 4) for a in aucs],
            "mean_auc": round(float(np.mean(aucs)), 4),
            "std_auc": round(float(np.std(aucs, ddof=1)), 4),
        }

    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    for name, build in builders.items():
        model = build().fit(X_train, y_train)
        if isinstance(model, _LightGBM):
            predictions["lightgbm_uncalibrated"] = model.raw(X_hold)
            model.iso = fit_isotonic(oof[name], y_train)
        predictions[name] = model.predict(X_hold)
        fitted[name] = model

    boot: dict[str, Any] = {
        "iterations": cfg.bootstrap_iterations,
        "confidence": cfg.confidence,
        "seed": seed,
    }
    holdout = {name: discrimination(y_hold, predictions[name], **boot) for name in builders}
    calibration = {
        name: calibration_summary(
            y_hold, p, bins=cfg.calibration_bins, low_risk_deciles=cfg.low_risk_deciles
        )
        for name, p in predictions.items()
    }
    tests = {
        f"{a}_vs_{b}": delong_test(y_hold, predictions[a], predictions[b])
        for a, b in combinations(builders, 2)
    }
    selection = cfg.champion_selection
    champion = select_champion(
        holdout,
        tests,
        alpha=selection.significance_level,
        min_gain=selection.min_auc_gain,
        simplicity_order=selection.simplicity_order,
    )

    fair_cfg = cfg.fairness
    protected_hold = data.protected.iloc[idx_hold]
    fairness: dict[str, dict[str, Any]] = {}
    for name in builders:
        approved = approve_at_rate(predictions[name], fair_cfg.approval_rate)
        fairness[name] = {
            attr: group_fairness(
                y_hold,
                approved,
                protected_hold[attr],
                air_threshold=fair_cfg.air_threshold,
                min_group_size=fair_cfg.group_floor(len(y_hold)),
            )
            for attr in data.protected.columns
        }

    attribute = cfg.lda.attribute
    if attribute == "auto":  # search where the champion is least fair
        champion_fairness = fairness[champion["model"]]
        attribute = min(champion_fairness, key=lambda a: champion_fairness[a]["min_air"])
    lda = less_discriminatory_search(
        X_train,
        y_train,
        X_hold,
        y_hold,
        data.protected.iloc[idx_train][attribute],
        protected_hold[attribute],
        approval_rate=fair_cfg.approval_rate,
        epsilons=cfg.lda.eg_epsilons,
        proxy_auc_threshold=cfg.lda.proxy_auc_threshold,
        air_threshold=fair_cfg.air_threshold,
        min_group_size=fair_cfg.group_floor(len(y_hold)),
        baseline_scores=predictions[champion["model"]],
        baseline_name=f"Champion: {champion['label']}",
    )
    lda["attribute"] = attribute
    lda["recommendation"] = recommend_lda(lda, cfg.lda.max_auc_loss)

    metrics = {
        "dataset": data.name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_version": cfg.version,
        "design": {
            "rows": len(data.y),
            "train_rows": len(y_train),
            "holdout_rows": len(y_hold),
            "cv_folds": cfg.cv_folds,
            "holdout_fraction": cfg.holdout_fraction,
            "seed": seed,
            "default_rate": round(float(data.y.mean()), 4),
            "features": list(data.X.columns),
            "protected_attributes": list(data.protected.columns),
            "notes": data.notes,
        },
        "models": MODEL_LABELS,
        "cv": cv,
        "holdout": holdout,
        "calibration": calibration,
        "delong": tests,
        "champion": champion,
        "fairness": {
            "approval_rate": fair_cfg.approval_rate,
            "min_group_size": fair_cfg.group_floor(len(y_hold)),
            "by_model": fairness,
        },
        "lda": lda,
        "shap_top_features": _shap_top(fitted["lightgbm"], X_hold, seed),
    }
    return LaneAResult(metrics=metrics, y_holdout=y_hold, predictions=predictions)
