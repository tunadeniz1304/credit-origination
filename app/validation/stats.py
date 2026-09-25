"""Statistical tests for model validation.

* :func:`delong_test` — DeLong, DeLong & Clarke-Pearson (1988) test for two
  correlated ROC AUCs, using the midrank formulation of Sun & Xu (2014).
* :func:`bootstrap_auc_ci` — stratified percentile bootstrap CI for the AUC.
* :func:`calibration_deciles`, :func:`expected_calibration_error`,
  :func:`hosmer_lemeshow` — calibration diagnostics; the low-risk deciles are
  reported separately because under-estimation there is the costly error for
  a lender (approved applicants look safer than they are).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ discrimination
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1  # average 1-based rank of the tie block
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_components(
    y: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AUCs and the DeLong structural components for ``k`` score vectors.

    ``scores`` has shape ``(k, n)``; returns ``(aucs, v10, v01)`` where ``v10``
    has shape ``(k, m)`` (one placement per positive) and ``v01`` ``(k, n0)``.
    """
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n0 = pos.shape[1], neg.shape[1]
    if m == 0 or n0 == 0:
        raise ValueError("both classes are required for an AUC")
    k = scores.shape[0]
    aucs = np.empty(k)
    v10 = np.empty((k, m))
    v01 = np.empty((k, n0))
    for r in range(k):
        tx, ty = _midrank(pos[r]), _midrank(neg[r])
        tz = _midrank(np.concatenate([pos[r], neg[r]]))
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n0)
        v10[r] = (tz[:m] - tx) / n0
        v01[r] = 1.0 - (tz[m:] - ty) / m
    return aucs, v10, v01


def delong_covariance(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aucs, v10, v01 = delong_components(y, scores)
    m, n0 = v10.shape[1], v01.shape[1]
    s10 = np.atleast_2d(np.cov(v10))
    s01 = np.atleast_2d(np.cov(v01))
    return aucs, s10 / m + s01 / n0


def delong_test(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, Any]:
    """Two-sided DeLong test of ``AUC(a) == AUC(b)`` on the same observations."""
    aucs, cov = delong_covariance(y, np.vstack([scores_a, scores_b]))
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = diff / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    half = float(stats.norm.ppf(0.975) * np.sqrt(max(var, 0.0)))
    return {
        "auc_a": round(float(aucs[0]), 5),
        "auc_b": round(float(aucs[1]), 5),
        "auc_diff": round(diff, 5),
        "diff_ci": [round(diff - half, 5), round(diff + half, 5)],
        "z": round(float(z), 4),
        "p_value": float(f"{p:.6g}"),
    }


def delong_auc_ci(y: np.ndarray, scores: np.ndarray, confidence: float = 0.95) -> list[float]:
    aucs, cov = delong_covariance(y, np.atleast_2d(scores))
    half = float(stats.norm.ppf(0.5 + confidence / 2) * np.sqrt(cov[0, 0]))
    return [round(float(aucs[0]) - half, 5), round(float(aucs[0]) + half, 5)]


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    aucs, _v10, _v01 = delong_components(y, p)
    return float(aucs[0])


def bootstrap_auc_ci(
    y: np.ndarray,
    p: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> list[float]:
    """Stratified percentile bootstrap (positives and negatives resampled apart)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = np.empty(iterations)
    for b in range(iterations):
        idx = np.concatenate(
            [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg))]
        )
        samples[b] = roc_auc_score(y[idx], p[idx])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return [round(float(lo), 5), round(float(hi), 5)]


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    # Evaluate only at real thresholds (end of each tie block), never inside a tie.
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]
    return float(np.max(np.abs(cum_bad[ends] - cum_good[ends])))


def discrimination(
    y: np.ndarray, p: np.ndarray, *, iterations: int, confidence: float, seed: int
) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, p))
    ci = bootstrap_auc_ci(y, p, iterations=iterations, confidence=confidence, seed=seed)
    return {
        "auc": round(auc, 4),
        "auc_ci": ci,
        "auc_ci_delong": delong_auc_ci(y, p, confidence),
        "gini": round(2 * auc - 1, 4),
        "gini_ci": [round(2 * ci[0] - 1, 4), round(2 * ci[1] - 1, 4)],
        "ks": round(ks_statistic(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, p)), 5),
        "default_rate": round(float(y.mean()), 4),
        "n": len(y),
    }


# ------------------------------------------------------------------ calibration
def calibration_deciles(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    """(Near) equal-count bins ordered from the lowest to the highest predicted PD.

    Rows with the same PD always fall in the same bin: a bin is assigned from the
    *average* rank of the tie block, so a calibrated model with few distinct PDs
    (isotonic steps) is not split arbitrarily by row order. Bins can therefore be
    uneven, and a bin swallowed by a large tie block is empty and omitted.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    ranks = stats.rankdata(p, method="average")
    assigned = np.minimum(((ranks - 1) * bins / len(p)).astype(int), bins - 1)
    rows = []
    for index in range(1, bins + 1):
        chunk = np.flatnonzero(assigned == index - 1)
        if len(chunk) == 0:
            continue
        predicted = float(p[chunk].mean())
        observed = float(y[chunk].mean())
        rows.append(
            {
                "decile": index,
                "n": len(chunk),
                "pd_min": round(float(p[chunk].min()), 5),
                "pd_max": round(float(p[chunk].max()), 5),
                "predicted": round(predicted, 5),
                "observed": round(observed, 5),
                "defaults": int(y[chunk].sum()),
                "ratio_observed_to_predicted": round(observed / predicted, 3)
                if predicted > 0
                else None,
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = calibration_deciles(y, p, bins)
    total = sum(r["n"] for r in table)
    return round(sum(r["n"] / total * abs(r["observed"] - r["predicted"]) for r in table), 5)


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    table = calibration_deciles(y, p, bins)
    statistic = 0.0
    for r in table:
        expected = r["predicted"] * r["n"]
        variance = expected * (1 - r["predicted"])
        if variance > 0:
            statistic += (r["defaults"] - expected) ** 2 / variance
    dof = max(len(table) - 2, 1)
    return {
        "statistic": round(statistic, 3),
        "dof": dof,
        "p_value": float(f"{stats.chi2.sf(statistic, dof):.6g}"),
    }


def calibration_summary(
    y: np.ndarray, p: np.ndarray, *, bins: int, low_risk_deciles: int
) -> dict[str, Any]:
    table = calibration_deciles(y, p, bins)
    low = table[:low_risk_deciles]
    low_n = sum(r["n"] for r in low)
    low_pred = sum(r["predicted"] * r["n"] for r in low) / max(low_n, 1)
    low_obs = sum(r["observed"] * r["n"] for r in low) / max(low_n, 1)
    return {
        "deciles": table,
        "ece": expected_calibration_error(y, p, bins),
        "hosmer_lemeshow": hosmer_lemeshow(y, p, bins),
        "mean_predicted": round(float(np.mean(p)), 5),
        "observed_rate": round(float(np.mean(y)), 5),
        "low_risk": {
            "deciles": low_risk_deciles,
            "predicted": round(low_pred, 5),
            "observed": round(low_obs, 5),
            "ratio_observed_to_predicted": round(low_obs / low_pred, 3) if low_pred else None,
        },
    }
