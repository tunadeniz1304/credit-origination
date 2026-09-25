"""Lane B: anchor the production model to real public data.

1. Map the Taiwan set onto the platform's bureau-behaviour features.
2. Record the real default curve per delinquency band (the anchors).
3. Train the bureau behaviour sub-score on real defaults.
4. Generate the synthetic population with the anchored generator and compare
   its per-band default curve with the real one (tolerance from
   ``rules/validation.yaml``).
5. Retrain the production PD model, scorecard and challenger on it and check
   the PD level and low-risk calibration against the anchors.
6. Measure the policy cut-off table (v1 and v2 thresholds: share auto-approved /
   referred / auto-declined and the realised bad rate of the auto-approved book)
   on the anchored synthetic test population.

The anchoring *imposes* the PD level from a real proxy curve (next-month card
default); it does not validate a 12-month 90+DPD loan PD.

Writes ``artifacts/validation/lane_b.json`` and the model artifacts; then run
``scripts/run_validation.py --report-only`` to refresh the report.

Usage: python scripts/run_lane_b.py [--rows 100000] [--fixture] [--policy-only]
[--v1-model-dir DIR]

``--policy-only`` keeps the trained models and the anchors: it regenerates the
(deterministic) synthetic population and refreshes only the production
calibration and the policy table (which records the SHA-256 of the scored model
files and the thresholds read from ``rules/policy_v2.yaml``). ``--v1-model-dir``
points at the retired ``pd_lgbm_v1`` artifact (``pd_lgbm_v1.txt`` + ``.meta.json``; removed from the
tree in commit 569ca14, recover with ``git show 569ca14^:artifacts/models/...``)
to add the historical row "v1 policy on the v1 model".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.core.rules import load_validation  # noqa: E402
from app.decisioning import public_mapping as pm  # noqa: E402
from app.validation.datasets import FIXTURE_PATH, dataset_path, load_frame  # noqa: E402

MODELS = ROOT / "artifacts" / "models"
# Thresholds of the retired rules/policy_v1.yaml (renamed to policy_v2 in b81b771).
POLICY_V1 = {"auto_approve_max_pd": 0.05, "auto_decline_min_pd": 0.20}


def artifact_sha256(path: Path) -> str:
    """SHA-256 of a text artifact with LF line endings (as committed, see .gitattributes).

    The digest therefore does not depend on the checkout's line-ending conversion.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def model_fingerprint(version: str, model_dir: Path = MODELS) -> dict[str, str]:
    """Digest of every file the production PD model is loaded from."""
    files = [model_dir / f"{version}.txt", model_dir / f"{version}.meta.json"]
    return {f"artifacts/models/{f.name}": artifact_sha256(f) for f in files}


def policy_row(
    pd_scores: np.ndarray, y: np.ndarray, approve_max: float, decline_min: float
) -> dict:
    """PD-threshold outcome of one policy (rule-based referrals/declines not applied)."""
    approve = pd_scores <= approve_max
    decline = pd_scores >= decline_min
    grey = ~approve & ~decline

    def rate(mask: np.ndarray) -> float | None:
        return round(float(y[mask].mean()), 4) if mask.any() else None

    return {
        "auto_approve_max_pd": approve_max,
        "auto_decline_min_pd": decline_min,
        "share_auto_approve": round(float(approve.mean()), 4),
        "share_referred": round(float(grey.mean()), 4),
        "share_auto_decline": round(float(decline.mean()), 4),
        "bad_rate_auto_approved": rate(approve),
        "bad_rate_referred": rate(grey),
        "bad_rate_auto_declined": rate(decline),
        "mean_pd_auto_approved": (
            round(float(pd_scores[approve].mean()), 4) if approve.any() else None
        ),
    }


def policy_cutoffs(test, pd_v2: np.ndarray, version: str, v1_model_dir: Path | None) -> dict:
    """Cut-off table of the v1 and v2 policies on the anchored synthetic test set."""
    from app.core.rules import load_policy_file

    y = test["target"].to_numpy()
    policy = load_policy_file()
    decision = policy.decision
    rows = [
        {
            "policy": "policy_v2",
            "model": version,
            **policy_row(pd_v2, y, decision.auto_approve_max_pd, decision.auto_decline_min_pd),
        },
        {
            "policy": "policy_v1 thresholds",
            "model": version,
            **policy_row(
                pd_v2, y, POLICY_V1["auto_approve_max_pd"], POLICY_V1["auto_decline_min_pd"]
            ),
        },
    ]
    if v1_model_dir is not None:
        import lightgbm as lgb

        from app.decisioning.models import PDModel

        meta = json.loads((v1_model_dir / "pd_lgbm_v1.meta.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(v1_model_dir / "pd_lgbm_v1.txt"))
        v1 = PDModel(booster, meta)
        rows.append(
            {
                "policy": "policy_v1",
                "model": f"{v1.version} (retired; git history)",
                **policy_row(
                    v1.predict_batch(test),
                    y,
                    POLICY_V1["auto_approve_max_pd"],
                    POLICY_V1["auto_decline_min_pd"],
                ),
            }
        )
    return {
        "population": "anchored synthetic test set (application_month > 20)",
        "test_rows": len(y),
        "observed_default_rate": round(float(y.mean()), 4),
        # Provenance: the table is valid only for these model files and thresholds.
        "model": version,
        "model_sha256": model_fingerprint(version),
        "thresholds": {
            "policy_v2": {
                "source": "rules/policy_v2.yaml",
                "version": policy.version,
                "auto_approve_max_pd": decision.auto_approve_max_pd,
                "auto_decline_min_pd": decision.auto_decline_min_pd,
            },
            "policy_v1": {"source": "retired rules/policy_v1.yaml", **POLICY_V1},
        },
        "note": (
            "Measured on the anchored synthetic population, whose PD level is imposed from a real "
            "proxy curve (next-month card default), not validated. Only the PD thresholds are "
            "applied; rule-based referrals and declines (DSR, fraud, documents) are not."
        ),
        "rows": rows,
    }


def production_checks(
    synthetic, real_curve: dict, cfg, bands: list[int], v1_model_dir: Path | None
) -> tuple[dict, dict]:
    """PD level / low-risk calibration and policy table of the trained production model."""
    from app.decisioning.features import MODEL_FEATURES
    from app.decisioning.models import _load
    from app.validation.stats import calibration_summary

    _load.cache_clear()
    models = _load(str(MODELS))
    test = synthetic[synthetic.application_month > 20]
    pd_test = models.pd_model.predict_batch(test[list(MODEL_FEATURES)])
    y_test = test["target"].to_numpy()
    calib = calibration_summary(
        y_test, pd_test, bins=cfg.calibration_bins, low_risk_deciles=cfg.low_risk_deciles
    )
    test_bureau = (test["bureau_hit"] == 1).to_numpy()
    band = pm.delinquency_band(np.ceil(test["max_dpd_24m"].to_numpy() / pm.DAYS_PER_MONTH), bands)
    per_band = []
    for b in bands:
        mask = test_bureau & (band == b)
        if not mask.any():
            continue
        per_band.append(
            {
                "band": str(b),
                "real_default_rate": real_curve[str(b)]["default_rate"],
                "mean_predicted_pd": round(float(pd_test[mask].mean()), 4),
                "n": int(mask.sum()),
            }
        )
    tolerance = cfg.lane_b.low_risk_tolerance_ratio
    low_deciles = calib["deciles"][: cfg.low_risk_deciles]
    ratio = calib["low_risk"]["ratio_observed_to_predicted"]
    production = {
        "model": models.pd_model.version,
        "test_rows": len(y_test),
        "ece": calib["ece"],
        "hosmer_lemeshow": calib["hosmer_lemeshow"],
        "low_risk": calib["low_risk"],
        "low_risk_within_tolerance": ratio is not None and abs(ratio - 1) <= tolerance,
        "low_risk_deciles_within_tolerance": all(
            d["ratio_observed_to_predicted"] is not None
            and abs(d["ratio_observed_to_predicted"] - 1) <= tolerance
            for d in low_deciles
        ),
        "tolerance_ratio": tolerance,
        "deciles": calib["deciles"],
        "per_band_level": per_band,
        "note": (
            "Internal consistency only: the synthetic outcomes are drawn from the same anchored "
            "curve the model is trained on, so a match shows the level was imposed correctly, "
            "not that it is right for a 12-month 90+DPD loan PD."
        ),
    }
    return production, policy_cutoffs(test, pd_test, models.pd_model.version, v1_model_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="keep models and anchors; refresh production calibration and the policy table",
    )
    parser.add_argument("--v1-model-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_validation()
    bands = cfg.lane_b.delinquency_bands
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if args.policy_only:
        from app.decisioning.training import SEED, generate_dataset

        result = json.loads(pm.LANE_B_PATH.read_text(encoding="utf-8"))
        real_curve = result["anchors"]["delinquency_band_default"]
        synthetic = generate_dataset(args.rows, SEED)
        result["production_calibration"], result["policy_cutoffs"] = production_checks(
            synthetic, real_curve, cfg, bands, args.v1_model_dir
        )
        pm.LANE_B_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        print(json.dumps(result["policy_cutoffs"], indent=1))
        return 0
    source = FIXTURE_PATH if args.fixture else dataset_path("uci_taiwan")
    public = pm.map_public(load_frame(source))

    real_curve = pm.default_curve(public["delay_months"], public["default"], bands)
    util_bins = np.quantile(public["utilisation"], np.linspace(0, 1, 6))
    util_curve = []
    for lo, hi in pairwise(util_bins):
        mask = (public["utilisation"] >= lo) & (public["utilisation"] <= hi)
        util_curve.append(
            {
                "utilisation": [round(float(lo), 3), round(float(hi), 3)],
                "default_rate": round(float(public.loc[mask, "default"].mean()), 4),
            }
        )
    anchors = {
        "source": "fixture" if args.fixture else "uci_taiwan (full)",
        "rows": len(public),
        "overall_default_rate": round(float(public["default"].mean()), 4),
        "delinquency_band_default": real_curve,
    }
    result: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_version": cfg.version,
        "mapping": pm.MAPPING_TABLE,
        "anchors": anchors,
        "real_utilisation_curve": util_curve,
    }
    pm.LANE_B_PATH.parent.mkdir(parents=True, exist_ok=True)
    pm.LANE_B_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    pm.lane_b_anchors.cache_clear()

    behaviour = pm.train_behaviour_score(
        public, seed=cfg.seed, holdout_fraction=cfg.holdout_fraction, out_dir=MODELS
    )
    result["behaviour_score"] = behaviour["metrics"] | {"version": behaviour["version"]}

    from app.decisioning.training import SEED, generate_dataset, train_all

    synthetic = generate_dataset(args.rows, SEED)
    has_bureau = synthetic["bureau_hit"] == 1
    synth_curve = pm.default_curve(
        np.ceil(synthetic.loc[has_bureau, "max_dpd_24m"] / pm.DAYS_PER_MONTH),
        synthetic.loc[has_bureau, "target"],
        bands,
    )
    tol = cfg.lane_b.band_tolerance_abs
    comparison = []
    for band in map(str, bands):
        real, synth = real_curve[band], synth_curve[band]
        gap = abs(real["default_rate"] - synth["default_rate"])
        comparison.append(
            {
                "band": band,
                "real_default_rate": real["default_rate"],
                "synthetic_default_rate": synth["default_rate"],
                "abs_gap": round(gap, 4),
                "within_tolerance": gap <= tol,
                "real_share": real["share"],
                "synthetic_share": synth["share"],
                "synthetic_n": synth["n"],
            }
        )
    result["generator_check"] = {
        "tolerance_abs": tol,
        "bands": comparison,
        "all_within_tolerance": all(c["within_tolerance"] for c in comparison),
        "synthetic_default_rate": round(float(synthetic["target"].mean()), 4),
        "note": (
            "Default rate per band is anchored to the real curve; band shares describe the "
            "synthetic Turkish applicant mix and intentionally differ from a credit-card book."
        ),
    }

    output = train_all(synthetic, MODELS)
    (MODELS / "metrics.json").write_text(
        json.dumps(
            {"rows": len(synthetic), "models": output.paths, "metrics": output.metrics},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    # PD level / low-risk calibration and policy table of the retrained production model.
    result["production_calibration"], result["policy_cutoffs"] = production_checks(
        synthetic, real_curve, cfg, bands, args.v1_model_dir
    )
    pm.LANE_B_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "mapping"}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
