"""Synthetic training data and model training (PD, scorecard, challenger).

``generate_dataset`` draws a seeded population from the same persona mix as
the mock providers, with realistic correlations (income, debt service,
bureau score, arrears, cash-flow behaviour, tenure) and a latent default
process; the target is *90+ days past due within 12 months*. Protected
attributes (gender, age band, province) are generated for fairness
monitoring only and are never model inputs.

``train_all`` fits:

* a monotone-constrained LightGBM PD model, time-based split
  (train months 1–16, calibration 17–20, test 21–24) with isotonic calibration;
* an ``optbinning`` WoE/IV logistic scorecard scaled to 300–900 points;
* a logistic-regression challenger (EBM when ``interpret`` is installed);

and reports AUC, Gini, KS, Brier and a calibration table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.decisioning.features import MODEL_FEATURES, MONOTONE, annuity_factor

SEED = 20260924
REFERENCE_RATE = 0.45
PERSONA_MIX = {
    "temiz": 0.45,
    "gri": 0.20,
    "asiri_borclu": 0.12,
    "gecikmeli": 0.08,
    "ince_dosya": 0.10,
    "serbest": 0.05,
}
PROVINCES = ("İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Diyarbakır", "Samsun", "Konya")


def _uniform(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + (hi - lo) * rng.random(len(lo))


def generate_dataset(n: int = 50_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    personas = rng.choice(list(PERSONA_MIX), size=n, p=list(PERSONA_MIX.values()))
    p = pd.Series(personas)

    def by(
        mapping: dict[str, tuple[float, float]], default: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = p.map(lambda k: mapping.get(k, default)[0]).to_numpy(float)
        hi = p.map(lambda k: mapping.get(k, default)[1]).to_numpy(float)
        return lo, hi

    gender = rng.choice(["K", "E"], size=n, p=[0.46, 0.54])
    age = rng.integers(21, 65, size=n)
    province = rng.choice(PROVINCES, size=n, p=[0.30, 0.14, 0.11, 0.08, 0.08, 0.10, 0.09, 0.10])
    income = rng.lognormal(np.log(42_000), 0.45, size=n)
    income *= np.where(gender == "K", 0.94, 1.0)  # realistic pay gap (monitoring only)
    income *= np.where(np.isin(province, ["İstanbul", "Ankara", "İzmir"]), 1.12, 0.95)
    income = np.clip(income, 17_000, 400_000)

    bureau_hit = np.where(p == "ince_dosya", 0, (rng.random(n) > 0.03).astype(int))
    score_lo, score_hi = by(
        {
            "temiz": (1420, 1780),
            "gri": (1060, 1240),
            "asiri_borclu": (1150, 1380),
            "gecikmeli": (700, 1000),
            "serbest": (1280, 1620),
        },
        (1300, 1600),
    )
    score = np.round(_uniform(rng, score_lo, score_hi) + rng.normal(0, 40, n))
    score = np.clip(score, 1, 1900).astype(float)
    score[bureau_hit == 0] = np.nan

    loans_lo, loans_hi = by(
        {
            "temiz": (0, 2),
            "gri": (1, 3),
            "asiri_borclu": (3, 5),
            "gecikmeli": (2, 4),
            "ince_dosya": (0, 0),
        },
        (0, 2),
    )
    active_loans = np.round(_uniform(rng, loans_lo, loans_hi)).astype(int)
    edsr_lo, edsr_hi = by(
        {
            "temiz": (0.03, 0.15),
            "gri": (0.10, 0.22),
            "asiri_borclu": (0.20, 0.32),
            "gecikmeli": (0.18, 0.34),
            "ince_dosya": (0, 0),
        },
        (0.05, 0.15),
    )
    existing_dsr = _uniform(rng, edsr_lo, edsr_hi)
    util_lo, util_hi = by(
        {
            "temiz": (0.05, 0.45),
            "gri": (0.5, 0.8),
            "asiri_borclu": (0.75, 0.98),
            "gecikmeli": (0.7, 0.98),
            "ince_dosya": (0, 0),
        },
        (0.1, 0.4),
    )
    utilisation = np.clip(_uniform(rng, util_lo, util_hi), 0, 1)
    delinq = np.where(
        p == "gecikmeli",
        rng.integers(2, 6, n),
        np.where(p == "gri", rng.binomial(1, 0.35, n), rng.binomial(1, 0.04, n)),
    )
    max_dpd = np.where(
        delinq > 0, np.where(p == "gecikmeli", rng.integers(30, 90, n), rng.integers(5, 30, n)), 0
    )
    inq_lo, inq_hi = by(
        {"temiz": (0, 2), "gri": (2, 4), "asiri_borclu": (3, 6), "gecikmeli": (4, 8)}, (0, 2)
    )
    inquiries = np.round(_uniform(rng, inq_lo, inq_hi)).astype(int)
    emp_lo, emp_hi = by(
        {
            "temiz": (36, 180),
            "gri": (8, 24),
            "gecikmeli": (6, 30),
            "ince_dosya": (24, 60),
            "serbest": (48, 200),
        },
        (24, 150),
    )
    employment = np.round(_uniform(rng, emp_lo, emp_hi)).astype(int)

    cv_lo, cv_hi = by(
        {"gri": (0.05, 0.12), "gecikmeli": (0.05, 0.14), "serbest": (0.12, 0.30)}, (0.008, 0.03)
    )
    income_cv = _uniform(rng, cv_lo, cv_hi)
    neg_lo, neg_hi = by({"gri": (5, 90), "asiri_borclu": (10, 80), "gecikmeli": (40, 200)}, (0, 0))
    neg_days = np.round(_uniform(rng, neg_lo, neg_hi) * (rng.random(n) < 0.9))
    nsf = np.where(p == "gecikmeli", rng.integers(1, 6, n), rng.binomial(1, 0.02, n))
    gamble = np.where(
        p == "gecikmeli",
        rng.uniform(0.0, 0.08, n),
        np.where(rng.random(n) < 0.03, rng.uniform(0.01, 0.15, n), 0.0),
    )
    sav_lo, sav_hi = by(
        {
            "temiz": (0.15, 0.40),
            "ince_dosya": (0.25, 0.45),
            "gri": (0.0, 0.10),
            "asiri_borclu": (-0.05, 0.05),
            "gecikmeli": (-0.10, 0.03),
        },
        (0.05, 0.3),
    )
    savings = _uniform(rng, sav_lo, sav_hi)
    bal_lo, bal_hi = by(
        {
            "temiz": (1.0, 3.5),
            "ince_dosya": (1.5, 3.5),
            "gri": (0.05, 0.6),
            "asiri_borclu": (0.05, 0.4),
            "gecikmeli": (0.0, 0.3),
        },
        (0.5, 2.5),
    )
    balance_ratio = _uniform(rng, bal_lo, bal_hi)

    term = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.12, 0.25, 0.33, 0.18, 0.12])
    amount = income * rng.uniform(1.0, 7.0, n)
    factor = np.array([annuity_factor(int(t), REFERENCE_RATE) for t in term])
    dsr = existing_dsr + amount * factor / income
    loan_to_income = amount / (income * 12)
    month = rng.integers(1, 25, size=n)
    inquiries = inquiries + (month > 20) * rng.binomial(1, 0.3, n)  # mild recent drift

    s = np.nan_to_num(score, nan=1250.0)
    logit = (
        -4.35
        - 0.0022 * (s - 1300)
        + np.where(bureau_hit == 0, 0.20, 0.0)
        + 3.0 * np.maximum(dsr - 0.35, 0)
        + 1.0 * np.minimum(dsr, 0.35)
        + 0.15 * delinq
        + 0.005 * max_dpd
        + 0.06 * inquiries
        + 0.7 * (utilisation - 0.5)
        + 0.05 * active_loans
        + 0.6 * loan_to_income
        + 0.005 * (term - 36)
        - 0.30 * (np.log(income) - np.log(42_000))
        - 0.004 * np.minimum(employment, 120)
        + 3.0 * income_cv
        + 0.003 * neg_days
        + 0.10 * nsf
        + 5.0 * gamble
        - 1.2 * savings
        - 0.20 * np.minimum(balance_ratio, 3)
        # Non-linear interactions a tree ensemble can learn but a linear model cannot.
        + 2.5 * np.maximum(dsr - 0.40, 0) * (savings < 0.05)
        + 4.0 * income_cv * (bureau_hit == 0)
        + 0.6 * ((utilisation > 0.85) & (inquiries >= 4))
        + rng.normal(0, 1.35, n)  # unobserved heterogeneity (life events)
    )
    pd_true = 1 / (1 + np.exp(-logit))
    target = (rng.random(n) < pd_true).astype(int)

    age_band = pd.cut(
        age, bins=[0, 29, 39, 49, 120], labels=["21-29", "30-39", "40-49", "50+"]
    ).astype(str)
    return pd.DataFrame(
        {
            "bureau_score": score,
            "bureau_hit": bureau_hit,
            "delinquency_count_24m": delinq,
            "max_dpd_24m": max_dpd,
            "inquiries_6m": inquiries,
            "active_loans": active_loans,
            "bureau_utilisation": np.round(utilisation, 4),
            "dsr": np.round(dsr, 4),
            "loan_to_income": np.round(loan_to_income, 4),
            "term_months": term,
            "log_income": np.round(np.log(income), 4),
            "employment_months": employment,
            "income_cv": np.round(income_cv, 4),
            "negative_balance_days": neg_days,
            "nsf_count": nsf,
            "gambling_share": np.round(gamble, 4),
            "savings_rate": np.round(savings, 4),
            "avg_balance_to_income": np.round(balance_ratio, 4),
            "persona": personas,
            "gender": gender,
            "age": age,
            "age_band": age_band,
            "province": province,
            "application_month": month,
            "requested_amount": np.round(amount, -2),
            "target": target,
        }
    )


# ------------------------------------------------------------------ metrics
def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "predicted": round(float(p[mask].mean()), 4),
                "observed": round(float(y[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return rows


def evaluate_scores(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    auc = float(roc_auc_score(y, p))
    return {
        "auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "default_rate": round(float(y.mean()), 4),
        "n": len(y),
    }


def distribution_bins(values: pd.Series, bins: int = 10) -> dict[str, list[float]]:
    """Quantile bin edges + reference shares for PSI drift monitoring."""
    clean = values.dropna().to_numpy(float)
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
    counts, _ = np.histogram(clean, bins=edges)
    shares = counts / max(counts.sum(), 1)
    return {
        "edges": [round(float(e), 6) for e in edges],
        "shares": [round(float(s), 6) for s in shares],
    }


# ------------------------------------------------------------------ training
@dataclass
class TrainingOutput:
    metrics: dict[str, Any]
    paths: dict[str, str]


def _splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df.application_month <= 16]
    calib = df[(df.application_month > 16) & (df.application_month <= 20)]
    test = df[df.application_month > 20]
    return train, calib, test


def train_pd_model(df: pd.DataFrame, out_dir: Path, version: str = "pd_lgbm_v1") -> dict[str, Any]:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    train, calib, test = _splits(df)
    features = list(MODEL_FEATURES)
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "monotone_constraints": [MONOTONE[f] for f in features],
        "monotone_constraints_method": "advanced",
        "verbose": -1,
        "seed": SEED,
        "deterministic": True,
        "num_threads": 1,
    }
    dtrain = lgb.Dataset(train[features], label=train.target)
    dvalid = lgb.Dataset(calib[features], label=calib.target, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    raw_calib = booster.predict(calib[features], num_iteration=booster.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0005, y_max=0.9995)
    iso.fit(raw_calib, calib.target)
    raw_test = booster.predict(test[features], num_iteration=booster.best_iteration)
    p_test = iso.predict(raw_test)
    metrics = evaluate_scores(test.target.to_numpy(), p_test)
    metrics["raw_auc"] = evaluate_scores(test.target.to_numpy(), raw_test)["auc"]
    metrics["calibration"] = calibration_table(test.target.to_numpy(), p_test)
    metrics["best_iteration"] = int(booster.best_iteration)
    importance = dict(zip(features, booster.feature_importance("gain").tolist(), strict=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    # model_to_string + Python I/O: LightGBM's native writer fails on non-ASCII paths.
    (out_dir / f"{version}.txt").write_text(
        booster.model_to_string(num_iteration=booster.best_iteration), encoding="utf-8"
    )
    meta = {
        "version": version,
        "kind": "lightgbm_monotone",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "monotone_constraints": {f: MONOTONE[f] for f in features},
        "calibration": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        "metrics": metrics,
        "feature_importance_gain": {k: round(v, 2) for k, v in importance.items()},
        "training": {
            "rows": len(df),
            "train_rows": len(train),
            "calibration_rows": len(calib),
            "test_rows": len(test),
            "split": "time-based: months 1-16 train, 17-20 calibration, 21-24 test",
            "target": "90+ DPD within 12 months",
            "seed": SEED,
        },
        "reference_distribution": {f: distribution_bins(train[f]) for f in features},
    }
    (out_dir / f"{version}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return meta


def train_scorecard(
    df: pd.DataFrame, out_dir: Path, version: str = "scorecard_woe_v1"
) -> dict[str, Any]:
    import joblib
    from optbinning import BinningProcess, Scorecard
    from sklearn.linear_model import LogisticRegression

    train, _calib, test = _splits(df)
    features = list(MODEL_FEATURES)
    # optbinning's monotonic_trend refers to the event rate: PD rises with +1 features.
    trends = {
        f: ("ascending" if MONOTONE[f] > 0 else "descending" if MONOTONE[f] < 0 else "auto")
        for f in features
    }
    binning = BinningProcess(
        variable_names=features,
        binning_fit_params={f: {"monotonic_trend": trends[f], "max_n_bins": 6} for f in features},
        selection_criteria={"iv": {"min": 0.01}},
    )
    scorecard = Scorecard(
        binning_process=binning,
        estimator=LogisticRegression(max_iter=1000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": 40, "odds": 50, "scorecard_points": 600},
    )
    scorecard.fit(train[features], train.target)
    test_scores = scorecard.score(test[features])
    p_test = scorecard.predict_proba(test[features])[:, 1]
    metrics = evaluate_scores(test.target.to_numpy(), p_test)
    iv = scorecard.binning_process_.summary()[["name", "iv", "selected"]]
    table = scorecard.table(style="summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorecard, out_dir / f"{version}.pkl", compress=3)
    meta = {
        "version": version,
        "kind": "optbinning_woe_logistic",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "scaling": {
            "method": "pdo_odds",
            "pdo": 40,
            "odds": 50,
            "base_points": 600,
            "range": [300, 900],
        },
        "metrics": metrics,
        "score_distribution": {
            "p05": round(float(np.percentile(test_scores, 5)), 1),
            "p50": round(float(np.percentile(test_scores, 50)), 1),
            "p95": round(float(np.percentile(test_scores, 95)), 1),
        },
        "information_value": {r["name"]: round(float(r["iv"]), 4) for _, r in iv.iterrows()},
        "selected": [r["name"] for _, r in iv.iterrows() if bool(r["selected"])],
        "max_points": {
            str(var): round(float(group["Points"].max()), 2)
            for var, group in table.groupby("Variable")
        },
    }
    (out_dir / f"{version}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return meta


def train_challenger(
    df: pd.DataFrame, out_dir: Path, version: str = "challenger_lr_v1"
) -> dict[str, Any]:
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, _calib, test = _splits(df)
    features = list(MODEL_FEATURES)
    kind = "logistic_regression"
    try:  # pragma: no cover - optional extra
        from interpret.glassbox import ExplainableBoostingClassifier

        model: Any = ExplainableBoostingClassifier(random_state=SEED)
        kind = "ebm"
        version = version.replace("lr", "ebm")
    except ImportError:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5),
        )
    model.fit(train[features], train.target)
    p_test = model.predict_proba(test[features])[:, 1]
    metrics = evaluate_scores(test.target.to_numpy(), p_test)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{version}.pkl", compress=3)
    meta = {
        "version": version,
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "metrics": metrics,
    }
    (out_dir / f"{version}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return meta


def train_all(df: pd.DataFrame, out_dir: Path) -> TrainingOutput:
    pd_meta = train_pd_model(df, out_dir)
    sc_meta = train_scorecard(df, out_dir)
    ch_meta = train_challenger(df, out_dir)
    return TrainingOutput(
        metrics={
            "pd": pd_meta["metrics"],
            "scorecard": sc_meta["metrics"],
            "challenger": ch_meta["metrics"],
        },
        paths={
            "pd": pd_meta["version"],
            "scorecard": sc_meta["version"],
            "challenger": ch_meta["version"],
        },
    )
