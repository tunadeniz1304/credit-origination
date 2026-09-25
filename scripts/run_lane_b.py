"""Lane B: anchor the production model to real public data.

1. Map the Taiwan set onto the platform's bureau-behaviour features.
2. Record the real default curve per delinquency band (the anchors).
3. Train the bureau behaviour sub-score on real defaults.
4. Generate the synthetic population with the anchored generator and compare
   its per-band default curve with the real one (tolerance from
   ``rules/validation.yaml``).
5. Retrain the production PD model, scorecard and challenger on it and check
   the PD level and low-risk calibration against the real anchors.

Writes ``artifacts/validation/lane_b.json`` and the model artifacts; then run
``scripts/run_validation.py --report-only`` to refresh the report.

Usage: python scripts/run_lane_b.py [--rows 100000] [--fixture]
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    cfg = load_validation()
    bands = cfg.lane_b.delinquency_bands
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

    # PD level / low-risk calibration of the retrained production model.
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
    result["production_calibration"] = {
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
    }
    pm.LANE_B_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(json.dumps({k: v for k, v in result.items() if k != "mapping"}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
