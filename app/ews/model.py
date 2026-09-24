"""Early-warning system (EWS) for disbursed loans.

* :func:`simulate_behaviour` — deterministic synthetic monthly behaviour after
  disbursement (days past due, balance, salary credit, new bureau arrears,
  card utilisation) whose deterioration probability follows the origination
  PD.
* :func:`ews_model` — logistic regression trained on a seeded synthetic
  behaviour population, predicting 90+ DPD within the next six months
  (``lifelines`` survival models are an optional extra).
* :func:`watchlist_entry` — score + rule triggers (new arrears at the bureau,
  salary stopped, DPD ≥ 30).
"""

from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

EWS_FEATURES = (
    "max_dpd_3m",
    "salary_missed_3m",
    "new_bureau_delinquency",
    "utilisation",
    "origination_pd",
)
WATCHLIST_THRESHOLD = 0.20


def _rng(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))


def simulate_behaviour(
    application_id: str, amount: float, instalment: float, pd: float, months: int = 6
) -> list[dict[str, Any]]:
    rng = _rng(application_id)
    stressed = rng.random() < min(0.95, pd * 3)
    onset = rng.randint(2, max(2, months)) if stressed else months + 1
    balance = amount
    dpd = 0
    rows = []
    for month in range(1, months + 1):
        deteriorating = month >= onset
        salary = not (deteriorating and rng.random() < 0.6)
        if deteriorating and rng.random() < 0.7:
            dpd = min(dpd + 30, 120)
        elif dpd and rng.random() < 0.5:
            dpd = max(dpd - 30, 0)
        if dpd == 0:
            balance = max(balance - instalment * 0.7, 0.0)
        rows.append(
            {
                "month": month,
                "dpd": dpd,
                "balance": round(balance, 2),
                "salary_credited": salary,
                "new_bureau_delinquency": bool(deteriorating and rng.random() < 0.4),
                "utilisation": round(
                    min(1.0, (0.35 if not deteriorating else 0.8) + rng.uniform(-0.1, 0.15)), 3
                ),
            }
        )
    return rows


def features_from_history(history: list[dict[str, Any]], origination_pd: float) -> dict[str, float]:
    recent = history[-3:]
    return {
        "max_dpd_3m": float(max((r["dpd"] for r in recent), default=0)),
        "salary_missed_3m": float(sum(1 for r in recent if not r["salary_credited"])),
        "new_bureau_delinquency": float(any(r["new_bureau_delinquency"] for r in recent)),
        "utilisation": float(recent[-1]["utilisation"]) if recent else 0.0,
        "origination_pd": float(origination_pd),
    }


@lru_cache(maxsize=1)
def ews_model() -> LogisticRegression:
    rng = np.random.default_rng(7)
    n = 6000
    pd0 = rng.beta(1.2, 18, n)
    dpd = rng.choice([0, 30, 60, 90], n, p=[0.82, 0.1, 0.05, 0.03])
    missed = rng.binomial(3, np.clip(pd0 * 2, 0, 0.9))
    arrears = rng.binomial(1, np.clip(pd0 * 1.5, 0, 0.9))
    util = np.clip(rng.normal(0.45, 0.2, n) + arrears * 0.25, 0, 1)
    logit = -4.2 + 0.035 * dpd + 0.7 * missed + 1.1 * arrears + 1.5 * util + 8 * pd0
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    X = np.column_stack([dpd, missed, arrears, util, pd0])
    return LogisticRegression(max_iter=1000).fit(X, y)


def ews_score(features: dict[str, float]) -> float:
    row = np.array([[features[f] for f in EWS_FEATURES]])
    return float(ews_model().predict_proba(row)[0, 1])


def watchlist_entry(
    application_id: str, history: list[dict[str, Any]], origination_pd: float
) -> dict[str, Any]:
    features = features_from_history(history, origination_pd)
    score = ews_score(features)
    triggers = []
    if features["new_bureau_delinquency"]:
        triggers.append("KKB'de yeni gecikme kaydı")
    last_two = history[-2:]
    if len(last_two) == 2 and not any(r["salary_credited"] for r in last_two):
        triggers.append("Maaş yatışı kesildi (2 ay)")
    if features["max_dpd_3m"] >= 30:
        triggers.append(f"Taksit gecikmesi ({int(features['max_dpd_3m'])} gün)")
    return {
        "application_id": application_id,
        "ews_score": round(score, 4),
        "on_watchlist": score >= WATCHLIST_THRESHOLD or bool(triggers),
        "triggers": triggers,
        "features": features,
        "months_observed": len(history),
    }
