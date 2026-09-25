"""Fairness testing and less-discriminatory-alternative (LDA) search.

Protected attributes are never model inputs; they are used here only to
*measure* outcomes.

* Every comparison is made at the **same approval rate** — comparing a
  quantile cut-off of one model with the 0.5-threshold ``predict()`` of
  another (the v1 table) mixes a fairness change with an approval-volume
  change and is meaningless.
* Ties at the cut-off (an isotonic calibration maps many applicants to the
  same PD) are broken by a seeded random key, never by row order; lane A
  reports the AIR spread over several tie-break seeds.
* An attribute with fewer than two groups above the minimum size is **not
  testable**: its AIR and gaps are ``None`` (shown as n/a), not a perfect 1.0.
* Per group: selection (approval) rate, adverse impact ratio (AIR, the
  four-fifths rule: group rate / most favoured group rate ≥ 0.8) and
  equalised-odds gaps — the difference between groups in the approval rate of
  good payers (TPR) and of defaulters (FPR).
* LDA search: proxy weakening (dropping features that on their own separate
  the protected groups), ``fairlearn`` ``ExponentiatedGradient`` with a
  demographic-parity bound, and demographic-parity group thresholds (the
  post-processing that ``ThresholdOptimizer`` performs), all re-thresholded to
  the same approval rate. Group-specific thresholds use the protected
  attribute at decision time and carry a legal caveat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

AIR_THRESHOLD = 0.8
PROTECTED = ("gender", "age_band", "province")
GROUP_THRESHOLD_CAVEAT = (
    "Grup bazlı eşik, korunan özelliğin karar anında kullanılması demektir; ayrımcılık yasağı "
    "(Anayasa m.10, 6701 sayılı Kanun, AB/ABD mevzuatı) altında doğrudan ayrımcılık sayılabilir. "
    "Yalnızca ödünleşimi göstermek için raporlanır, üretimde kullanılmaz."
)


# ------------------------------------------------------------------ approval at a fixed rate
def approve_at_rate(pd_scores: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Approve exactly ``round(rate·n)`` applicants with the lowest PD.

    Applicants with the same PD at the cut-off are ordered by a random key drawn
    from ``seed``, so which of them is approved does not depend on row order
    (a stable sort would favour whoever happens to come first in the file).
    """
    scores = np.asarray(pd_scores, dtype=float)
    k = round(rate * len(scores))
    tie_break = np.random.default_rng(seed).random(len(scores))
    approved = np.zeros(len(scores), dtype=int)
    approved[np.lexsort((tie_break, scores))[:k]] = 1
    return approved


def group_thresholds_at_rate(
    pd_scores: np.ndarray, groups: pd.Series, rate: float, seed: int = 0
) -> np.ndarray:
    """Demographic-parity post-processing: every group approved at ``rate``."""
    scores = np.asarray(pd_scores, dtype=float)
    labels = np.asarray(groups)
    approved = np.zeros(len(scores), dtype=int)
    for group in np.unique(labels):
        idx = np.flatnonzero(labels == group)
        approved[idx] = approve_at_rate(scores[idx], rate, seed)
    return approved


# ------------------------------------------------------------------ group metrics
def adverse_impact(
    approved: np.ndarray, groups: pd.Series, min_group_size: int = 0
) -> dict[str, Any]:
    from fairlearn.metrics import MetricFrame, selection_rate

    approved = np.asarray(approved).astype(int)
    frame = MetricFrame(
        metrics=selection_rate, y_true=approved, y_pred=approved, sensitive_features=groups
    )
    sizes = pd.Series(np.asarray(groups)).value_counts()
    rates = {
        str(k): round(float(v), 4)
        for k, v in frame.by_group.items()
        if sizes.get(k, 0) >= min_group_size
    }
    top = max(rates.values()) if rates else 0.0
    air = {g: round(r / top, 4) if top else None for g, r in rates.items()}
    # With fewer than two comparable groups there is nothing to compare: not testable.
    testable = len(rates) >= 2
    worst = min((v for v in air.values() if v is not None), default=None) if testable else None
    return {
        "selection_rate": rates,
        "air": air if testable else {g: None for g in rates},
        "min_air": worst,
        "testable": testable,
        "passes_four_fifths": None if worst is None else worst >= AIR_THRESHOLD,
    }


def equalized_odds_gap(y_true: np.ndarray, approved: np.ndarray, groups: pd.Series) -> float:
    from fairlearn.metrics import equalized_odds_difference

    # "Positive" outcome for the applicant is approval of a good (non-defaulting) loan.
    return round(
        float(equalized_odds_difference(1 - y_true, approved, sensitive_features=groups)), 4
    )


def group_fairness(
    y_true: np.ndarray,
    approved: np.ndarray,
    groups: pd.Series,
    *,
    air_threshold: float = AIR_THRESHOLD,
    min_group_size: int = 0,
) -> dict[str, Any]:
    """Selection rate, AIR and equalised-odds rates per group."""
    y = np.asarray(y_true).astype(int)
    approved = np.asarray(approved).astype(int)
    labels = np.asarray(groups).astype(str)
    stats = adverse_impact(approved, pd.Series(labels), min_group_size)
    tpr: dict[str, float] = {}
    fpr: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for group in stats["selection_rate"]:
        mask = labels == group
        good, bad = mask & (y == 0), mask & (y == 1)
        sizes[group] = int(mask.sum())
        tpr[group] = round(float(approved[good].mean()), 4) if good.any() else float("nan")
        fpr[group] = round(float(approved[bad].mean()), 4) if bad.any() else float("nan")
    finite = [v for v in tpr.values() if not np.isnan(v)]
    finite_fpr = [v for v in fpr.values() if not np.isnan(v)]

    def gap(values: list[float]) -> float | None:
        return round(max(values) - min(values), 4) if len(values) >= 2 else None

    stats.update(
        {
            "group_size": sizes,
            "tpr_good_approved": tpr,
            "fpr_bad_approved": fpr,
            "tpr_gap": gap(finite) if stats["testable"] else None,
            "fpr_gap": gap(finite_fpr) if stats["testable"] else None,
            "passes_four_fifths": None
            if stats["min_air"] is None
            else stats["min_air"] >= air_threshold,
        }
    )
    return stats


def air_spread(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    groups: pd.Series,
    *,
    approval_rate: float,
    seeds: list[int],
    air_threshold: float = AIR_THRESHOLD,
    min_group_size: int = 0,
) -> dict[str, Any] | None:
    """Minimum AIR over several tie-break seeds (min / median / max, share passing).

    Tied PDs at the cut-off are approved in a seeded random order; the spread shows
    how much a fairness conclusion depends on that arbitrary choice. ``None`` when
    the attribute is not testable.
    """
    values = []
    for seed in seeds:
        stats = group_fairness(
            y_true,
            approve_at_rate(pd_scores, approval_rate, seed),
            groups,
            air_threshold=air_threshold,
            min_group_size=min_group_size,
        )
        if stats["min_air"] is None:
            return None
        values.append(float(stats["min_air"]))
    arr = np.asarray(values)
    return {
        "seeds": len(seeds),
        "min": round(float(arr.min()), 4),
        "median": round(float(np.median(arr)), 4),
        "max": round(float(arr.max()), 4),
        "share_passing": round(float((arr >= air_threshold).mean()), 4),
    }


def fairness_report(
    df: pd.DataFrame, pd_scores: np.ndarray, approve_cutoff: float
) -> dict[str, Any]:
    """Synthetic-population report at the policy PD cut-off (monitoring view)."""
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
        stats = group_fairness(y, approved, df[attr])
        stats["equalized_odds_difference"] = equalized_odds_gap(y, approved, df[attr])
        report["attributes"][attr] = stats
    report["alerts"] = [
        f"{attr}: AIR {stats['min_air']:.2f} < {AIR_THRESHOLD}"
        for attr, stats in report["attributes"].items()
        if stats["passes_four_fifths"] is False
    ]
    return report


# ------------------------------------------------------------------ LDA search
def proxy_strength(X: pd.DataFrame, groups: pd.Series) -> dict[str, float]:
    """How well each feature alone separates the protected groups (one-vs-rest AUC)."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(groups).astype(str)
    out: dict[str, float] = {}
    for column in X.columns:
        values = X[column].fillna(X[column].median()).to_numpy(float)
        best = 0.5
        for group in np.unique(labels):
            target = (labels == group).astype(int)
            if 0 < target.sum() < len(target):
                auc = float(roc_auc_score(target, values))
                best = max(best, auc, 1 - auc)
        out[column] = round(best, 4)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _logistic() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )


def _row(
    name: str,
    y: np.ndarray,
    scores: np.ndarray,
    approved: np.ndarray,
    groups: pd.Series,
    air_threshold: float,
    min_group_size: int,
    **extra: Any,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    stats = group_fairness(
        y, approved, groups, air_threshold=air_threshold, min_group_size=min_group_size
    )
    return {
        "model": name,
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "approval_rate": round(float(approved.mean()), 4),
        "bad_rate_approved": round(float(y[approved == 1].mean()), 4) if approved.any() else 0.0,
        "min_air": stats["min_air"],
        "selection_rate": stats["selection_rate"],
        "tpr_gap": stats["tpr_gap"],
        "fpr_gap": stats["fpr_gap"],
        "passes_four_fifths": stats["passes_four_fifths"],
        **extra,
    }


def less_discriminatory_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    a_train: pd.Series,
    a_test: pd.Series,
    *,
    approval_rate: float,
    epsilons: list[float],
    proxy_auc_threshold: float,
    air_threshold: float = AIR_THRESHOLD,
    min_group_size: int = 0,
    baseline_scores: np.ndarray | None = None,
    baseline_name: str = "Kısıtsız lojistik",
    alternatives: dict[str, np.ndarray] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Performance–fairness trade-off table, every row at ``approval_rate``.

    ``alternatives`` are hold-out scores of the other model families that were
    already trained (e.g. the scorecard when LightGBM is champion): they are
    genuine less discriminatory alternative candidates and are ranked with the rest.
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient

    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    common: dict[str, Any] = {"air_threshold": air_threshold, "min_group_size": min_group_size}
    rows: list[dict[str, Any]] = []

    base = _logistic().fit(X_train, y_train)
    lr_scores = base.predict_proba(X_test)[:, 1]
    if baseline_scores is not None:
        rows.append(
            _row(
                baseline_name,
                y_test,
                baseline_scores,
                approve_at_rate(baseline_scores, approval_rate, seed),
                a_test,
                kind="champion",
                **common,
            )
        )
    rows.append(
        _row(
            "Kısıtsız lojistik",
            y_test,
            lr_scores,
            approve_at_rate(lr_scores, approval_rate, seed),
            a_test,
            kind="baseline",
            **common,
        )
    )
    for family, scores in (alternatives or {}).items():
        rows.append(
            _row(
                family,
                y_test,
                scores,
                approve_at_rate(scores, approval_rate, seed),
                a_test,
                kind="model_family",
                family=family,
                **common,
            )
        )

    proxies = proxy_strength(X_train, a_train)
    dropped = [f for f, auc in proxies.items() if auc >= proxy_auc_threshold]
    if not dropped:  # still show the effect of weakening the single strongest proxy
        dropped = list(proxies)[:1]
    kept = [c for c in X_train.columns if c not in dropped]
    weakened = _logistic().fit(X_train[kept], y_train)
    weak_scores = weakened.predict_proba(X_test[kept])[:, 1]
    rows.append(
        _row(
            "Proxy zayıflatılmış lojistik",
            y_test,
            weak_scores,
            approve_at_rate(weak_scores, approval_rate, seed),
            a_test,
            kind="proxy_removal",
            dropped_features=dropped,
            **common,
        )
    )

    # The reduction re-weights samples, which a Pipeline does not forward: fit it on the
    # imputed + standardised matrix with a bare logistic regression.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    xtr = scaler.transform(imputer.transform(X_train))
    xte = scaler.transform(imputer.transform(X_test))
    for eps in epsilons:
        mitigator = ExponentiatedGradient(
            LogisticRegression(max_iter=2000, C=0.5), DemographicParity(difference_bound=eps)
        )
        mitigator.fit(xtr, y_train, sensitive_features=np.asarray(a_train).astype(str))
        # Weighted mixture of the reduction's probabilistic predictors (a ranking score);
        # ``_pmf_predict`` mixes hard 0/1 predictions and would understate the AUC.
        eg_scores = np.zeros(len(xte))
        for weight, predictor in zip(mitigator.weights_, mitigator.predictors_, strict=True):
            if weight > 0:
                eg_scores += weight * predictor.predict_proba(xte)[:, 1]
        rows.append(
            _row(
                f"ExponentiatedGradient (demografik eşitlik, ε={eps})",
                y_test,
                eg_scores,
                approve_at_rate(eg_scores, approval_rate, seed),
                a_test,
                kind="exponentiated_gradient",
                epsilon=eps,
                **common,
            )
        )

    reference = baseline_scores if baseline_scores is not None else lr_scores
    rows.append(
        _row(
            "Grup bazlı eşik (demografik eşitlik, ThresholdOptimizer yaklaşımı)",
            y_test,
            reference,
            group_thresholds_at_rate(reference, a_test, approval_rate, seed),
            a_test,
            kind="group_threshold",
            legal_caveat=GROUP_THRESHOLD_CAVEAT,
            **common,
        )
    )

    # fairlearn's own ThresholdOptimizer picks its operating point by an objective,
    # not by volume: reported for reference, excluded from the same-rate comparison.
    optimizer = ThresholdOptimizer(
        estimator=base,
        constraints="demographic_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    optimizer.fit(X_train, y_train, sensitive_features=np.asarray(a_train).astype(str))
    predicted_default = optimizer.predict(
        X_test, sensitive_features=np.asarray(a_test).astype(str), random_state=0
    )
    to_row = _row(
        "fairlearn ThresholdOptimizer (kendi çalışma noktası)",
        y_test,
        lr_scores,
        (1 - np.asarray(predicted_default)).astype(int),
        a_test,
        kind="threshold_optimizer_reference",
        legal_caveat=GROUP_THRESHOLD_CAVEAT,
        **common,
    )
    to_row["comparable"] = False
    return {
        "approval_rate": approval_rate,
        "proxy_strength": dict(list(proxies.items())[:8]),
        "rows": rows,
        "reference_rows": [to_row],
    }


def recommend_lda(table: dict[str, Any], max_auc_loss: float) -> dict[str, Any]:
    """Pick the fairest non-group-threshold row within the allowed AUC loss.

    Candidates include the other already-trained model families (``model_family``
    rows), not only the purpose-built mitigations.
    """
    rows = table["rows"]
    anchor = rows[0]
    base = {"anchor": anchor["model"], "max_auc_loss": max_auc_loss}
    if anchor["min_air"] is None:
        return {
            **base,
            "recommended": None,
            "kind": None,
            "testable": False,
            "reason": "Korunan özellik test edilemiyor (karşılaştırılabilir ikinci grup yok).",
        }
    candidates = [
        r
        for r in rows[1:]
        if r["kind"] != "group_threshold"
        and r["min_air"] is not None
        and anchor["auc"] - r["auc"] <= max_auc_loss
    ]
    better = [r for r in candidates if r["min_air"] > anchor["min_air"]]
    if not better:
        return {
            **base,
            "recommended": None,
            "kind": None,
            "reason": (
                f"İzin verilen AUC kaybı ({max_auc_loss:.3f}) içinde {anchor['model']} modelinden "
                "daha yüksek AIR veren alternatif yok."
            ),
        }
    best = max(better, key=lambda r: (r["min_air"], r["auc"]))
    return {
        **base,
        "recommended": best["model"],
        "kind": best["kind"],
        "min_air_from": anchor["min_air"],
        "min_air_to": best["min_air"],
        "auc_loss": round(anchor["auc"] - best["auc"], 4),
        "reason": (
            f"AIR {anchor['min_air']:.3f} → {best['min_air']:.3f}, AUC kaybı "
            f"{anchor['auc'] - best['auc']:.4f} (sınır {max_auc_loss:.3f})."
        ),
    }


def less_discriminatory_alternatives(
    df: pd.DataFrame, features: list[str], attribute: str = "gender", approval_rate: float = 0.7
) -> list[dict[str, Any]]:
    """Synthetic-population wrapper: time split, all rows at the same approval rate."""
    from app.core.rules import load_validation

    cfg = load_validation()
    train = df[df.application_month <= 18]
    test = df[df.application_month > 18]
    table = less_discriminatory_search(
        train[features],
        train.target.to_numpy(),
        test[features],
        test.target.to_numpy(),
        train[attribute],
        test[attribute],
        approval_rate=approval_rate,
        epsilons=cfg.lda.eg_epsilons,
        proxy_auc_threshold=cfg.lda.proxy_auc_threshold,
        air_threshold=cfg.fairness.air_threshold,
    )
    return list(table["rows"])


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
