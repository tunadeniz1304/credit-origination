"""Lane A: validate the modelling methodology on real public credit data.

Reads ``data/external/<set>/data.csv`` (run ``scripts/fetch_public_credit_data.py``
first), writes ``artifacts/validation/<set>/metrics.json`` (committed, so the
app and tests work offline), the plots under ``docs/img/validation/`` and
``docs/VALIDATION_REPORT.md``.

Usage::

    python scripts/run_validation.py [--sets uci_taiwan german_credit] [--fixture]

``--fixture`` runs the Taiwan lane on the committed 3,000-row sample instead of
the full data (for a quick offline check; the committed report uses the full sets).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.core.rules import load_validation  # noqa: E402
from app.validation.datasets import (  # noqa: E402
    FIXTURE_PATH,
    available,
    dataset_path,
    load_frame,
    prepare,
)
from app.validation.lane_a import run_lane_a  # noqa: E402
from app.validation.report import MODEL_NAMES, dataset_section  # noqa: E402

OUT = ROOT / "artifacts" / "validation"
IMG = ROOT / "docs" / "img" / "validation"
REPORT = ROOT / "docs" / "VALIDATION_REPORT.md"


def plots(name: str, y: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    from app.validation.stats import calibration_deciles

    IMG.mkdir(parents=True, exist_ok=True)
    colours = {
        "lightgbm": "#1f3b57",
        "lightgbm_uncalibrated": "#9aa5b1",
        "logistic": "#c05621",
        "scorecard": "#2f855a",
    }
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    top = 0.0
    for key, p in predictions.items():
        table = calibration_deciles(y, p)
        xs, ys = [r["predicted"] for r in table], [r["observed"] for r in table]
        top = max(top, *xs, *ys)
        ax.plot(xs, ys, marker="o", ms=3, color=colours[key], label=MODEL_NAMES[key])
    ax.plot([0, top * 1.05], [0, top * 1.05], linestyle="--", color="#cbd2d9", label="Perfect")
    ax.set_xlabel("Predicted PD (decile mean)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration — {name} hold-out")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(IMG / f"{name}_calibration.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    for key, p in predictions.items():
        if key == "lightgbm_uncalibrated":
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colours[key], label=MODEL_NAMES[key])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#cbd2d9")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {name} hold-out")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(IMG / f"{name}_roc.png")
    plt.close(fig)


def summary_lines(results: dict[str, dict]) -> list[str]:
    lines = [
        "| Dataset | Champion | Champion AUC [95% CI] | LightGBM vs LR (ΔAUC, DeLong p) "
        "| Low-risk obs/pred | Worst AIR at 70% approval |",
        "|---|---|---|---|---|---|",
    ]
    alerts = []
    for name, m in results.items():
        champion = m["champion"]["model"]
        test = m["delong"].get("lightgbm_vs_logistic", {})
        hold = m["holdout"][champion]
        low = m["calibration"][champion]["low_risk"]["ratio_observed_to_predicted"]
        attrs = m["fairness"]["by_model"][champion]
        worst = min(attrs, key=lambda a: attrs[a]["min_air"])
        air = attrs[worst]["min_air"]
        lines.append(
            f"| {name} | {MODEL_NAMES[champion]} | {hold['auc']:.4f} [{hold['auc_ci'][0]:.3f}, "
            f"{hold['auc_ci'][1]:.3f}] | {test.get('auc_diff', 0):+.4f}, "
            f"p={test.get('p_value', float('nan')):.3g} | {low} | {air:.3f} ({worst}) |"
        )
        if not attrs[worst]["passes_four_fifths"]:
            alerts.append(
                f"- **Alert — {name}:** the champion fails the four-fifths rule for `{worst}` "
                f"(min AIR {air:.3f}); no less discriminatory alternative stays within the allowed "
                "AUC loss, so the finding goes to the model risk committee (see the LDA table)."
            )
        if low is not None and abs(low - 1) > 0.25:
            alerts.append(
                f"- **Alert — {name}:** low-risk deciles observed/predicted = {low} "
                "(under-estimation of risk where approvals concentrate)."
            )
    return [*lines, "", *alerts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="*", default=["uci_taiwan", "german_credit"])
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument(
        "--report-only", action="store_true", help="re-render the report from committed metrics"
    )
    args = parser.parse_args()
    cfg = load_validation()
    results: dict[str, dict] = {}
    for name in args.sets:
        if args.report_only:
            path = OUT / name / "metrics.json"
            if path.is_file():
                results[name] = json.loads(path.read_text(encoding="utf-8"))
            continue
        if args.fixture and name == "uci_taiwan":
            frame = load_frame(FIXTURE_PATH)
        elif available(name):
            frame = load_frame(dataset_path(name))
        else:
            print(f"{name}: data/external missing, run scripts/fetch_public_credit_data.py")
            continue
        data = prepare(name, frame, cfg.fairness)
        result = run_lane_a(data, cfg)
        result.metrics["source"] = "fixture" if args.fixture and name == "uci_taiwan" else "full"
        target = OUT / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        plots(name, result.y_holdout, result.predictions)
        results[name] = result.metrics
        champion = result.metrics["champion"]
        print(f"{name}: champion={champion['model']} :: " + " | ".join(champion["steps"]))

    if not results:
        return 1
    lines = [
        "# Validation Report — methodology on real public credit data",
        "",
        "Generated by `scripts/run_validation.py` from `artifacts/validation/<set>/metrics.json` "
        "(parameters: `rules/validation.yaml`). Data provenance, licences and checksums: "
        "[`docs/DATA.md`](DATA.md).",
        "",
        "**What this shows.** The production modelling recipe — monotone LightGBM with isotonic "
        "calibration, logistic regression, optbinning WoE scorecard, SHAP, fairness and LDA search — "
        "run unchanged on real, labelled default data with each set's own variables. **What it does "
        "not show.** The production model's Turkish features (KKB score, DSR, open-banking cash flow) "
        "are synthetic; its numbers on synthetic data are not evidence of real-world performance.",
        "",
        "## Summary",
        "",
        *summary_lines(results),
        "",
    ]
    for name, metrics in results.items():
        lines += dataset_section(name, metrics, "img/validation")
    lane_b = OUT / "lane_b.json"
    if lane_b.is_file():
        from app.validation.report import lane_b_section

        lines += lane_b_section(json.loads(lane_b.read_text(encoding="utf-8")))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"report: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
