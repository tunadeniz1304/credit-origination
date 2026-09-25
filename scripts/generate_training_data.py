"""Generate the seeded synthetic training population (50k applications).

Usage: python scripts/generate_training_data.py [--rows 100000] [--out data/generated/training.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.decisioning.training import SEED, generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="data/generated/training.csv")
    args = parser.parse_args()
    df = generate_dataset(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out} (default rate {df.target.mean():.2%})")
    print(df.groupby("persona").target.agg(["mean", "count"]).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
