"""Lane B: semantic mapping of public credit data onto platform features.

The platform's bureau (KKB) features have public counterparts in the UCI
Taiwan set. The mapping is deliberately conservative and documented in
``docs/DATA.md``:

=========================  ==================================  =====================================
Platform feature            Public source (Taiwan)              Transformation
=========================  ==================================  =====================================
``delay_months``            ``PAY_0 … PAY_6``                   max(0, status) over the window (months)
``delinquent_months``       ``PAY_0 … PAY_6``                   number of months with status > 0
``utilisation``             ``BILL_AMT1 / LIMIT_BAL``           clipped to [0, 1.5]
``age``                     ``AGE``                             unchanged (eligibility only)
=========================  ==================================  =====================================

On the platform side ``delay_months = ceil(max_dpd_24m / 30)`` and
``delinquent_months = delinquency_count_24m``, both capped at the public
ranges; the public window (6 months) is shorter than KKB's 24 months, so the
platform features are a superset of the public signal.

The **bureau behaviour sub-score** is a monotone LightGBM on these three
behaviour features, trained and isotonically calibrated on *real* Taiwan
defaults; it enters the production PD model as the feature
``bureau_behavior_score`` (the real-data default probability of the
applicant's bureau behaviour).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import PROJECT_ROOT

BEHAVIOUR_FEATURES: tuple[str, ...] = ("delay_months", "delinquent_months", "utilisation")
BEHAVIOUR_MONOTONE = {"delay_months": 1, "delinquent_months": 1, "utilisation": 1}
MAX_DELAY_MONTHS = 8  # the public PAY_* scale stops at 8+ months
MAX_DELINQUENT_MONTHS = 6  # six observed months in the public set
MAX_UTILISATION = 1.5
DAYS_PER_MONTH = 30
BEHAVIOUR_VERSION = "bureau_behavior_v1"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
LANE_B_PATH = PROJECT_ROOT / "artifacts" / "validation" / "lane_b.json"

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

MAPPING_TABLE: list[dict[str, str]] = [
    {
        "platform": "max_dpd_24m → delay_months",
        "public": "PAY_0…PAY_6 (repayment status, months of delay)",
        "transform": "platform: ceil(max_dpd_24m / 30), capped at 8; public: max(0, max PAY_x)",
        "rationale": "Both measure the worst arrears in months; KKB reports days past due.",
    },
    {
        "platform": "delinquency_count_24m → delinquent_months",
        "public": "PAY_0…PAY_6",
        "transform": "platform: count capped at 6; public: number of months with PAY_x > 0",
        "rationale": "Frequency of arrears; the public window is 6 months (platform: 24).",
    },
    {
        "platform": "bureau_utilisation → utilisation",
        "public": "BILL_AMT1 / LIMIT_BAL",
        "transform": "ratio clipped to [0, 1.5]",
        "rationale": "Revolving balance over limit is the standard bureau utilisation measure.",
    },
    {
        "platform": "age",
        "public": "AGE",
        "transform": "unchanged",
        "rationale": "Used only for eligibility rules and fairness monitoring, never as a model input.",
    },
]


def delinquency_band(delay_months: Any, bands: list[int]) -> Any:
    """0, 1, 2, … with the last configured band open-ended (e.g. 3 = 3+)."""
    return np.clip(np.asarray(delay_months, dtype=float), 0, max(bands)).astype(int)


def map_public(frame: pd.DataFrame) -> pd.DataFrame:
    """Taiwan rows → platform behaviour features (+ ``age`` and ``default``)."""
    status = frame[PAY_STATUS].clip(lower=0)
    limit = frame["LIMIT_BAL"].clip(lower=1)
    out = pd.DataFrame(
        {
            "delay_months": status.max(axis=1).clip(upper=MAX_DELAY_MONTHS),
            "delinquent_months": (status > 0).sum(axis=1),
            "utilisation": (frame["BILL_AMT1"].clip(lower=0) / limit).clip(0, MAX_UTILISATION),
            "age": frame["AGE"],
        },
        index=frame.index,
    ).astype(float)
    if "default" in frame:
        out["default"] = frame["default"].astype(int)
    return out


def map_platform(values: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    """Platform bureau features (snapshot or training frame) → behaviour features."""
    frame = pd.DataFrame([values]) if isinstance(values, dict) else values
    dpd = pd.to_numeric(frame["max_dpd_24m"], errors="coerce").fillna(0.0)
    count = pd.to_numeric(frame["delinquency_count_24m"], errors="coerce").fillna(0.0)
    util = pd.to_numeric(frame["bureau_utilisation"], errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "delay_months": np.ceil(dpd / DAYS_PER_MONTH).clip(0, MAX_DELAY_MONTHS),
            "delinquent_months": count.clip(0, MAX_DELINQUENT_MONTHS),
            "utilisation": util.clip(0, MAX_UTILISATION),
        },
        index=frame.index,
    ).astype(float)


def default_curve(delay_months: Any, target: Any, bands: list[int]) -> dict[str, dict[str, float]]:
    """Default rate and share per delinquency band."""
    band = delinquency_band(delay_months, bands)
    y = np.asarray(target, dtype=float)
    out: dict[str, dict[str, float]] = {}
    for b in bands:
        mask = band == b
        out[str(b)] = {
            "default_rate": round(float(y[mask].mean()), 4) if mask.any() else float("nan"),
            "share": round(float(mask.mean()), 4),
            "n": int(mask.sum()),
        }
    return out


# ------------------------------------------------------------------ behaviour sub-score
@dataclass
class BehaviourScore:
    booster: Any
    cal_x: np.ndarray
    cal_y: np.ndarray
    meta: dict[str, Any]

    def score_frame(self, behaviour: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(behaviour[list(BEHAVIOUR_FEATURES)])
        return np.clip(np.interp(raw, self.cal_x, self.cal_y), 0.0005, 0.9995)

    def score_platform(self, values: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        return self.score_frame(map_platform(values))


def train_behaviour_score(
    public: pd.DataFrame, *, seed: int, holdout_fraction: float, out_dir: Path = MODEL_DIR
) -> dict[str, Any]:
    """Fit on real defaults; isotonic calibration and metrics on a stratified hold-out."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from app.decisioning.training import fit_isotonic, fit_lightgbm

    X = public[list(BEHAVIOUR_FEATURES)]
    y = public["default"].to_numpy(int)
    x_fit, x_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=seed
    )
    x_tr, x_cal, y_tr, y_cal = train_test_split(
        x_fit, y_fit, test_size=0.25, stratify=y_fit, random_state=seed
    )
    booster = fit_lightgbm(x_tr, y_tr, x_cal, y_cal, BEHAVIOUR_MONOTONE)
    iso = fit_isotonic(booster.predict(x_cal), y_cal)
    p_hold = np.asarray(iso.predict(booster.predict(x_hold)))
    meta = {
        "version": BEHAVIOUR_VERSION,
        "kind": "lightgbm_monotone_behaviour",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": list(BEHAVIOUR_FEATURES),
        "monotone_constraints": BEHAVIOUR_MONOTONE,
        "trained_on": "UCI Default of Credit Card Clients (real defaults), mapped features",
        "calibration": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        "metrics": {
            "holdout_auc": round(float(roc_auc_score(y_hold, p_hold)), 4),
            "holdout_rows": len(y_hold),
            "holdout_mean_pd": round(float(p_hold.mean()), 4),
            "holdout_default_rate": round(float(y_hold.mean()), 4),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{BEHAVIOUR_VERSION}.txt").write_text(
        booster.model_to_string(num_iteration=booster.best_iteration), encoding="utf-8"
    )
    (out_dir / f"{BEHAVIOUR_VERSION}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    load_behaviour_score.cache_clear()
    return meta


@lru_cache(maxsize=2)
def load_behaviour_score(directory: str = str(MODEL_DIR)) -> BehaviourScore:
    import lightgbm as lgb

    path = Path(directory)
    meta = json.loads((path / f"{BEHAVIOUR_VERSION}.meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_str=(path / f"{BEHAVIOUR_VERSION}.txt").read_text(encoding="utf-8"))
    return BehaviourScore(
        booster=booster,
        cal_x=np.asarray(meta["calibration"]["x"], dtype=float),
        cal_y=np.asarray(meta["calibration"]["y"], dtype=float),
        meta=meta,
    )


def behaviour_score_for(snapshot: dict[str, Any]) -> float | None:
    """Real-data behaviour PD for a decision snapshot (``None`` for thin files)."""
    if not snapshot.get("bureau_hit"):
        return None
    return round(float(load_behaviour_score().score_platform(snapshot)[0]), 5)


@lru_cache(maxsize=1)
def lane_b_anchors() -> dict[str, Any]:
    """Real-data anchors committed by ``scripts/run_lane_b.py``."""
    data: dict[str, Any] = json.loads(LANE_B_PATH.read_text(encoding="utf-8"))
    return data["anchors"]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
