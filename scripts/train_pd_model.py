"""Train the PD model, the WoE scorecard and the challenger; write artifacts.

Usage:
    python scripts/train_pd_model.py [--rows 100000] [--data data/generated/training.csv]
                                     [--out artifacts/models]

Outputs (``artifacts/models``): LightGBM booster + metadata (calibration,
metrics, reference distributions), pickled scorecard and challenger, a metrics
summary and ``docs/img/calibration.png``. Governance reports (fairness) are
produced by ``scripts/fairness_report.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.decisioning.training import SEED, generate_dataset, train_all  # noqa: E402


def _calibration_plot(meta_path: Path, target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = meta["metrics"]["calibration"]
    predicted = [r["predicted"] for r in rows]
    observed = [r["observed"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    top = max(predicted + observed) * 1.1
    ax.plot([0, top], [0, top], linestyle="--", color="#9aa5b1", label="Mükemmel kalibrasyon")
    ax.plot(predicted, observed, marker="o", color="#1f3b57", label="PD modeli (test)")
    ax.set_xlabel("Tahmin edilen PD")
    ax.set_ylabel("Gözlenen temerrüt oranı")
    ax.set_title("Kalibrasyon — pd_lgbm_v2 (zaman bazlı test seti)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--data", default="")
    parser.add_argument("--out", default="artifacts/models")
    args = parser.parse_args()
    if args.data and Path(args.data).is_file():
        df = pd.read_csv(args.data)
    else:
        df = generate_dataset(args.rows, SEED)
    out = ROOT / args.out
    result = train_all(df, out)
    summary = {"rows": len(df), "models": result.paths, "metrics": result.metrics}
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _calibration_plot(
        out / f"{result.paths['pd']}.meta.json", ROOT / "docs" / "img" / "calibration.png"
    )
    for name, metrics in result.metrics.items():
        print(
            f"{name:10s} AUC={metrics['auc']:.4f} Gini={metrics['gini']:.4f} "
            f"KS={metrics['ks']:.4f} Brier={metrics['brier']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
