"""Model registry: calibrated PD model, scorecard and shadow challenger.

Artifacts are produced by ``scripts/train_pd_model.py`` into
``artifacts/models``. When they are missing (fresh clone without artifacts),
a small model set is trained on the fly from the seeded generator so the
platform still starts. SHAP contributions come from ``shap.TreeExplainer`` on
the LightGBM margin (log-odds).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger
from app.decisioning.features import MODEL_FEATURES, model_vector

PD_VERSION = "pd_lgbm_v1"
SCORECARD_VERSION = "scorecard_woe_v1"
CHALLENGER_VERSION = "challenger_lr_v1"
SCORE_MIN, SCORE_MAX = 300, 900
_lock = threading.Lock()


@dataclass
class ModelOutput:
    pd: float
    raw_score: float
    shap: dict[str, float]
    base_value: float


class PDModel:
    """LightGBM booster + isotonic calibration."""

    def __init__(self, booster: Any, meta: dict[str, Any]) -> None:
        self.booster = booster
        self.meta = meta
        self.version: str = meta["version"]
        self.features: list[str] = meta["features"]
        self._cal_x = np.asarray(meta["calibration"]["x"], dtype=float)
        self._cal_y = np.asarray(meta["calibration"]["y"], dtype=float)
        self._explainer: Any | None = None

    def _frame(self, snapshot: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([model_vector(snapshot)], columns=list(MODEL_FEATURES))[self.features]

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self._cal_x, self._cal_y)

    def predict(self, snapshot: dict[str, Any], explain: bool = True) -> ModelOutput:
        frame = self._frame(snapshot)
        raw = float(self.booster.predict(frame)[0])
        pd_value = float(np.clip(self.calibrate(np.array([raw]))[0], 0.0005, 0.9995))
        shap_values: dict[str, float] = {}
        base = 0.0
        if explain:
            explainer = self.explainer()
            values = explainer.shap_values(frame)
            row = values[1][0] if isinstance(values, list) else values[0]
            shap_values = {f: round(float(v), 5) for f, v in zip(self.features, row, strict=True)}
            expected = explainer.expected_value
            base = float(
                expected[1]
                if isinstance(expected, list | np.ndarray) and np.size(expected) > 1
                else expected
            )
        return ModelOutput(
            pd=round(pd_value, 5),
            raw_score=round(raw, 5),
            shap=shap_values,
            base_value=round(base, 5),
        )

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(frame[self.features])
        return np.clip(self.calibrate(raw), 0.0005, 0.9995)

    def explainer(self) -> Any:
        if self._explainer is None:
            import shap

            with _lock:
                if self._explainer is None:
                    self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer


class ScorecardModel:
    """optbinning WoE scorecard scaled to 300–900 points."""

    def __init__(self, model: Any, meta: dict[str, Any]) -> None:
        self.model = model
        self.meta = meta
        self.version: str = meta["version"]
        self.max_points: dict[str, float] = meta.get("max_points", {})
        self._table = model.table(style="detailed")

    def score(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([model_vector(snapshot)], columns=list(MODEL_FEATURES))
        total = float(self.model.score(frame)[0])
        points: dict[str, float] = {}
        table = self._table
        binning = self.model.binning_process_
        for variable in table["Variable"].unique():
            optb = binning.get_binned_variable(variable)
            value = frame[variable].to_numpy()
            bin_index = optb.transform(value, metric="indices")[0]
            var_rows = table[table["Variable"] == variable].reset_index(drop=True)
            if 0 <= bin_index < len(var_rows):
                points[variable] = round(float(var_rows.loc[bin_index, "Points"]), 2)
        lost = {
            var: round(max(0.0, self.max_points.get(var, pts) - pts), 2)
            for var, pts in points.items()
        }
        return {
            "points": round(float(np.clip(total, SCORE_MIN, SCORE_MAX)), 1),
            "raw_points": round(total, 2),
            "feature_points": points,
            "points_lost": dict(sorted(lost.items(), key=lambda kv: kv[1], reverse=True)),
            "version": self.version,
        }


class ChallengerModel:
    def __init__(self, model: Any, meta: dict[str, Any]) -> None:
        self.model = model
        self.meta = meta
        self.version: str = meta["version"]

    def predict(self, snapshot: dict[str, Any]) -> float:
        frame = pd.DataFrame([model_vector(snapshot)], columns=list(MODEL_FEATURES))
        return round(float(self.model.predict_proba(frame)[:, 1][0]), 5)


@dataclass
class ModelSet:
    pd_model: PDModel
    scorecard: ScorecardModel
    challenger: ChallengerModel

    @property
    def versions(self) -> dict[str, str]:
        return {
            "pd": self.pd_model.version,
            "scorecard": self.scorecard.version,
            "challenger": self.challenger.version,
        }


def _ensure_artifacts(directory: Path) -> None:
    if (directory / f"{PD_VERSION}.txt").is_file() and (
        directory / f"{SCORECARD_VERSION}.pkl"
    ).is_file():
        return
    get_logger("decisioning.models").warning(
        "Model artifacts missing in %s: training a small fallback set", directory
    )
    from app.decisioning.training import generate_dataset, train_all

    train_all(generate_dataset(8000), directory)


@lru_cache(maxsize=2)
def _load(directory: str) -> ModelSet:
    import joblib
    import lightgbm as lgb

    path = Path(directory)
    _ensure_artifacts(path)
    pd_meta = json.loads((path / f"{PD_VERSION}.meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_str=(path / f"{PD_VERSION}.txt").read_text(encoding="utf-8"))
    sc_meta = json.loads((path / f"{SCORECARD_VERSION}.meta.json").read_text(encoding="utf-8"))
    ch_meta = json.loads((path / f"{CHALLENGER_VERSION}.meta.json").read_text(encoding="utf-8"))
    return ModelSet(
        pd_model=PDModel(booster, pd_meta),
        scorecard=ScorecardModel(joblib.load(path / f"{SCORECARD_VERSION}.pkl"), sc_meta),
        challenger=ChallengerModel(joblib.load(path / f"{CHALLENGER_VERSION}.pkl"), ch_meta),
    )


def get_models() -> ModelSet:
    return _load(str(get_settings().models_path))


def models_loaded() -> bool:
    return _load.cache_info().currsize > 0
