"""Committed real-data validation evidence (lane A / lane B) for governance.

``artifacts/validation/<set>/metrics.json`` is produced offline by
``scripts/run_validation.py`` and committed, so the application reads the
evidence without network or the full datasets.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.core.rules import load_validation

VALIDATION_DIR = PROJECT_ROOT / "artifacts" / "validation"
PRIMARY_SET = "uci_taiwan"
# Production model kind -> lane-A model family.
KIND_TO_FAMILY = {
    "lightgbm_monotone": "lightgbm",
    "logistic_regression": "logistic",
    "ebm": "logistic",
    "optbinning_woe_logistic": "scorecard",
}


@lru_cache(maxsize=8)
def _read(path: str, mtime: float) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def load_metrics(name: str) -> dict[str, Any] | None:
    path = VALIDATION_DIR / name / "metrics.json"
    if not path.is_file():
        return None
    return _read(str(path), path.stat().st_mtime)


def load_lane_b() -> dict[str, Any] | None:
    path = VALIDATION_DIR / "lane_b.json"
    if not path.is_file():
        return None
    return _read(str(path), path.stat().st_mtime)


def available_sets() -> list[str]:
    if not VALIDATION_DIR.is_dir():
        return []
    return sorted(p.parent.name for p in VALIDATION_DIR.glob("*/metrics.json"))


def pair_test(metrics: dict[str, Any], a: str, b: str) -> dict[str, Any] | None:
    """DeLong result for ``a`` vs ``b`` whatever order it was stored in."""
    tests = metrics.get("delong", {})
    if f"{a}_vs_{b}" in tests:
        return dict(tests[f"{a}_vs_{b}"])
    if f"{b}_vs_{a}" in tests:
        flipped = tests[f"{b}_vs_{a}"]
        lo, hi = flipped["diff_ci"]
        return {
            **flipped,
            "auc_a": flipped["auc_b"],
            "auc_b": flipped["auc_a"],
            "auc_diff": -flipped["auc_diff"],
            "diff_ci": [-hi, -lo],
            "z": -flipped["z"],
        }
    return None


def family_evidence(champion: str, challenger: str, name: str = PRIMARY_SET) -> dict[str, Any]:
    """Hold-out AUC with CIs, DeLong test and calibration for two model families."""
    metrics = load_metrics(name)
    if metrics is None:
        return {"available": False, "dataset": name}
    holdout, calibration = metrics["holdout"], metrics["calibration"]
    test = pair_test(metrics, challenger, champion)

    def view(family: str) -> dict[str, Any]:
        return {
            "family": family,
            "auc": holdout[family]["auc"],
            "auc_ci": holdout[family]["auc_ci"],
            "brier": holdout[family]["brier"],
            "ece": calibration[family]["ece"],
            "low_risk_ratio": calibration[family]["low_risk"]["ratio_observed_to_predicted"],
        }

    return {
        "available": True,
        "dataset": name,
        "champion": view(champion),
        "challenger": view(challenger),
        "auc_ci": {
            champion: holdout[champion]["auc_ci"],
            challenger: holdout[challenger]["auc_ci"],
        },
        "delong_p_value": test["p_value"] if test else None,
        "auc_diff_challenger_minus_champion": test["auc_diff"] if test else None,
        "lane_a_champion": metrics["champion"]["model"],
        "generated_at": metrics.get("generated_at"),
    }


def promotion_evidence(champion_kind: str, challenger_kind: str) -> dict[str, Any]:
    """Promotion is allowed only if the challenger is not significantly worse on real data."""
    champion = KIND_TO_FAMILY.get(champion_kind)
    challenger = KIND_TO_FAMILY.get(challenger_kind)
    if champion is None or challenger is None:
        return {"allowed": False, "reason": "model ailesi için gerçek veri kanıtı yok"}
    evidence = family_evidence(champion, challenger)
    if not evidence["available"]:
        return {"allowed": False, "reason": "gerçek veri doğrulama metrikleri bulunamadı"}
    alpha = load_validation().champion_selection.significance_level
    diff = evidence["auc_diff_challenger_minus_champion"]
    p = evidence["delong_p_value"]
    worse = diff is not None and p is not None and diff < 0 and p < alpha
    reason = (
        f"challenger gerçek veride anlamlı biçimde daha zayıf (ΔAUC={diff:+.4f}, DeLong p={p:.3g})"
        if worse
        else f"challenger gerçek veride champion'dan anlamlı biçimde zayıf değil (ΔAUC={diff:+.4f}, "
        f"DeLong p={p:.3g})"
    )
    return {"allowed": not worse, "reason": reason, "evidence": evidence}
