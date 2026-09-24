"""Application anomaly score with an IsolationForest.

The detector is fitted once per process on a seeded synthetic reference
population of normal applications (income, amount, term, age, amount/income)
and returns a score in ``[0, 1]`` where higher means more unusual. It is a
signal and a feature, never a decision on its own.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = ("log_income", "log_amount", "term_months", "age", "amount_to_income")
_SEED = 20260924


@lru_cache(maxsize=1)
def _detector() -> tuple[IsolationForest, float, float]:
    rng = np.random.default_rng(_SEED)
    n = 4000
    income = rng.lognormal(mean=np.log(42_000), sigma=0.45, size=n)
    amount = income * rng.uniform(1.0, 9.0, size=n)
    term = rng.choice([12, 18, 24, 36, 48, 60], size=n)
    age = rng.integers(21, 64, size=n)
    matrix = np.column_stack([np.log(income), np.log(amount), term, age, amount / income])
    model = IsolationForest(n_estimators=150, contamination="auto", random_state=_SEED)
    model.fit(matrix)
    raw = -model.score_samples(matrix)
    return model, float(np.percentile(raw, 1)), float(np.percentile(raw, 99.5))


def anomaly_score(*, income: float, amount: float, term_months: int, age: int) -> float:
    model, lo, hi = _detector()
    row = np.array(
        [
            [
                np.log(max(income, 1.0)),
                np.log(max(amount, 1.0)),
                term_months,
                age,
                amount / max(income, 1.0),
            ]
        ]
    )
    raw = float(-model.score_samples(row)[0])
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))
