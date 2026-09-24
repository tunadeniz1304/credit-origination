"""Fairness testing (Zest-style LDA search, simplified).

Protected / proxy attributes (gender, age band, province) are never model
inputs; they are used here only to *monitor* outcomes:

* ``fairlearn.MetricFrame`` selection rate per group, adverse impact ratio
  (AIR, four-fifths rule: min rate / max rate ≥ 0.8) and equalised-odds gaps;
* a "less discriminatory alternative" search: a demographic-parity
  constrained model (``ExponentiatedGradient``) and threshold variants, with
  the performance-fairness trade-off table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

AIR_THRESHOLD = 0.8
PROTECTED = ("gender", "age_band", "province")


def adverse_impact(approved: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    from fairlearn.metrics import MetricFrame, selection_rate

    frame = MetricFrame(
        metrics=selection_rate, y_true=approved, y_pred=approved, sensitive_features=groups
    )
    rates = {str(k): round(float(v), 4) for k, v in frame.by_group.items()}
    top = max(rates.values()) if rates else 0.0
    air = {g: round(r / top, 4) if top else None for g, r in rates.items()}
    worst = min((v for v in air.values() if v is not None), default=1.0)
    return {
        "selection_rate": rates,
        "air": air,
        "min_air": worst,
        "passes_four_fifths": worst >= AIR_THRESHOLD,
    }


def equalized_odds_gap(y_true: np.ndarray, approved: np.ndarray, groups: pd.Series) -> float:
    from fairlearn.metrics import equalized_odds_difference

    # "Positive" outcome for the applicant is approval of a good (non-defaulting) loan.
    return round(
        float(equalized_odds_difference(1 - y_true, approved, sensitive_features=groups)), 4
    )


def fairness_report(
    df: pd.DataFrame, pd_scores: np.ndarray, approve_cutoff: float
) -> dict[str, Any]:
    approved = (pd_scores <= approve_cutoff).astype(int)
    y = df["target"].to_numpy()
    report: dict[str, Any] = {
        "approve_cutoff": approve_cutoff,
        "overall_approval_rate": round(float(approved.mean()), 4),
        "attributes": {},
    }
    for attr in PROTECTED:
        if attr not in df:
            continue
        stats = adverse_impact(approved, df[attr])
        stats["equalized_odds_difference"] = equalized_odds_gap(y, approved, df[attr])
        report["attributes"][attr] = stats
    report["alerts"] = [
        f"{attr}: AIR {stats['min_air']:.2f} < {AIR_THRESHOLD}"
        for attr, stats in report["attributes"].items()
        if not stats["passes_four_fifths"]
    ]
    return report


def less_discriminatory_alternatives(
    df: pd.DataFrame, features: list[str], attribute: str = "gender"
) -> list[dict[str, Any]]:
    """Trade-off table: unconstrained vs demographic-parity constrained models."""
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train = df[df.application_month <= 18]
    test = df[df.application_month > 18]
    imputer = SimpleImputer(strategy="median").fit(train[features])
    xtr, xte = imputer.transform(train[features]), imputer.transform(test[features])
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    ytr, yte = train.target.to_numpy(), test.target.to_numpy()
    rows: list[dict[str, Any]] = []
    base = LogisticRegression(max_iter=2000).fit(xtr, ytr)
    p = base.predict_proba(xte)[:, 1]
    cutoff = float(np.quantile(p, 0.7))
    approved = (p <= cutoff).astype(int)
    rows.append(
        {
            "model": "Kısıtsız lojistik",
            "auc": round(float(roc_auc_score(yte, p)), 4),
            "approval_rate": round(float(approved.mean()), 4),
            "min_air": adverse_impact(approved, test[attribute])["min_air"],
        }
    )
    for eps in (0.05, 0.02):
        mitigator = ExponentiatedGradient(
            LogisticRegression(max_iter=2000), DemographicParity(difference_bound=eps)
        )
        mitigator.fit(xtr, ytr, sensitive_features=train[attribute])
        pred_default = mitigator.predict(xte)
        approved_c = (1 - pred_default).astype(int)
        proba = mitigator._pmf_predict(xte)[:, 1]
        rows.append(
            {
                "model": f"Demografik eşitlik kısıtlı (ε={eps})",
                "auc": round(float(roc_auc_score(yte, proba)), 4),
                "approval_rate": round(float(approved_c.mean()), 4),
                "min_air": adverse_impact(approved_c, test[attribute])["min_air"],
            }
        )
    return rows


def live_fairness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """AIR on live decisions (monitoring attributes captured with consent)."""
    if not rows:
        return {"decisions": 0, "attributes": {}}
    frame = pd.DataFrame(rows)
    out: dict[str, Any] = {"decisions": len(frame), "attributes": {}}
    for attr in ("gender", "age_band", "province"):
        if attr in frame and frame[attr].notna().sum() >= 2 and frame[attr].nunique() > 1:
            sub = frame[frame[attr].notna()]
            out["attributes"][attr] = adverse_impact(
                sub["approved"].astype(int).to_numpy(), sub[attr]
            )
    return out
