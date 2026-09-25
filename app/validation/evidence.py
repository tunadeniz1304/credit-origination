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
# Production model kind -> lane-A model family. A kind that is not listed has no real-data
# evidence of its own and is refused; it never borrows the evidence of another family.
KIND_TO_FAMILY = {
    "lightgbm_monotone": "lightgbm",
    "logistic_regression": "logistic",
    "optbinning_woe_logistic": "scorecard",
}
# Promotion is refused when the challenger is significantly worse on either basis.
BASES = ("out_of_fold", "holdout")
BASE_LABELS = {"out_of_fold": "katlama dışı", "holdout": "hold-out"}
EVIDENCE_SCOPE = (
    "Kanıt, model ailesi / modelleme tarifi içindir (kulvar A: aynı tarif gerçek halka açık "
    "veride). Üretim artefaktı sentetik veriyle eğitildiğinden artefakt özetine bağlı gerçek "
    "veri kanıtı mümkün değildir."
)


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


def pair_test(
    metrics: dict[str, Any], a: str, b: str, basis: str = "holdout"
) -> dict[str, Any] | None:
    """DeLong result for ``a`` vs ``b`` whatever order it was stored in.

    ``basis`` is ``"holdout"`` (``delong``) or ``"out_of_fold"`` (``delong_oof``, the
    pooled out-of-fold scores the champion was selected on).
    """
    tests = metrics.get("delong_oof" if basis == "out_of_fold" else "delong", {})
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
    test_oof = pair_test(metrics, challenger, champion, "out_of_fold")

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
        "oof_delong_p_value": test_oof["p_value"] if test_oof else None,
        "oof_auc_diff_challenger_minus_champion": test_oof["auc_diff"] if test_oof else None,
        "lane_a_champion": metrics["champion"]["model"],
        "generated_at": metrics.get("generated_at"),
    }


def promotion_evidence(champion_kind: str, challenger_kind: str) -> dict[str, Any]:
    """Promotion gate on the committed real-data evidence (rule: ``rules/validation.yaml``).

    * both kinds must map to a lane-A family — an unknown family is refused;
    * at least one real dataset must be available, and (``require_every_dataset``)
      every available dataset must hold evidence for both families;
    * the challenger must not be significantly worse than the champion (paired DeLong,
      ΔAUC < 0 and p < α) on **any** dataset on **either** basis: the pooled out-of-fold
      scores (the basis the lane-A champion is chosen on, ``delong_oof``) or the hold-out
      (``delong``). A small hold-out cannot overrule the out-of-fold evidence.

    The evidence is for the family / recipe, not for the production artifact.
    """
    cfg = load_validation()
    champion = KIND_TO_FAMILY.get(champion_kind)
    challenger = KIND_TO_FAMILY.get(challenger_kind)
    base: dict[str, Any] = {
        "evidence_scope": EVIDENCE_SCOPE,
        "rule": "not_worse_on_any_dataset",
        "bases": list(BASES),
    }
    if champion is None or challenger is None:
        unknown = [k for k in (champion_kind, challenger_kind) if k not in KIND_TO_FAMILY]
        return {
            **base,
            "allowed": False,
            "reason": f"model ailesi için gerçek veri kanıtı yok ({', '.join(unknown)}); "
            "başka bir ailenin kanıtı kullanılmaz",
        }
    names = available_sets()
    if not names:
        return {**base, "allowed": False, "reason": "gerçek veri doğrulama metrikleri bulunamadı"}
    alpha = cfg.champion_selection.significance_level
    per_set: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in names:
        metrics = load_metrics(name) or {}
        holdout = metrics.get("holdout", {})
        tests = {b: test for b in BASES if (test := pair_test(metrics, challenger, champion, b))}
        if champion not in holdout or challenger not in holdout or len(tests) < len(BASES):
            missing.append(name)
            continue
        row: dict[str, Any] = {"dataset": name}
        for b, test in tests.items():
            diff, p = float(test["auc_diff"]), float(test["p_value"])
            row[b] = {
                "auc_diff_challenger_minus_champion": diff,
                "delong_p_value": p,
                "significantly_worse": diff < 0 and p < alpha,
            }
        row["worse_on"] = [b for b in BASES if row[b]["significantly_worse"]]
        row["significantly_worse"] = bool(row["worse_on"])
        per_set.append(row)
    base |= {"families": {"champion": champion, "challenger": challenger}, "datasets": per_set}
    if not per_set or (missing and cfg.promotion.require_every_dataset):
        return {
            **base,
            "allowed": False,
            "reason": "aile için gerçek veri kanıtı eksik: " + ", ".join(missing or names),
        }
    worse = [d for d in per_set if d["significantly_worse"]]
    detail = "; ".join(
        f"{d['dataset']} "
        + ", ".join(
            f"{BASE_LABELS[b]} ΔAUC={d[b]['auc_diff_challenger_minus_champion']:+.4f} "
            f"(DeLong p={d[b]['delong_p_value']:.3g})"
            for b in BASES
        )
        for d in (worse or per_set)
    )
    reason = (
        f"challenger gerçek veride anlamlı biçimde daha zayıf ({detail})"
        if worse
        else f"challenger hiçbir gerçek veri setinde champion'dan anlamlı biçimde zayıf değil "
        f"({detail})"
    )
    return {
        **base,
        "allowed": not worse,
        "reason": f"{reason}. Aile düzeyinde kanıt; üretim artefaktına bağlı değildir.",
        "evidence": family_evidence(champion, challenger),
    }
