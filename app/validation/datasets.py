"""Public credit datasets: loading, own-feature preparation, protected attributes.

Lane A trains on each set's **own** variables. Protected attributes are split
off before modelling and used only for the fairness analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import PROJECT_ROOT
from app.core.rules import FairnessConfig

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "uci_taiwan_sample.csv"

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]


@dataclass
class PreparedData:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    protected: pd.DataFrame
    monotone: dict[str, int]
    notes: list[str] = field(default_factory=list)


def dataset_path(name: str) -> Path:
    return EXTERNAL_DIR / name / "data.csv"


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def available(name: str) -> bool:
    return dataset_path(name).is_file()


def age_band(age: pd.Series, cfg: FairnessConfig) -> pd.Series:
    return pd.cut(age, bins=cfg.age_bins, labels=cfg.age_labels, right=True).astype(str)


# ------------------------------------------------------------------ Taiwan
def taiwan_engineered(frame: pd.DataFrame) -> pd.DataFrame:
    """Ratios every lane-A model gets in addition to the raw variables."""
    delays = frame[PAY_STATUS].clip(lower=0)
    limit = frame["LIMIT_BAL"].clip(lower=1)
    bills = frame[BILL].clip(lower=0)
    return pd.DataFrame(
        {
            "max_delay_6m": delays.max(axis=1),
            "months_delayed_6m": (delays > 0).sum(axis=1),
            "utilisation": (bills["BILL_AMT1"] / limit).clip(0, 2),
            "payment_ratio": (frame[PAY_AMT].sum(axis=1) / bills.sum(axis=1).clip(lower=1)).clip(
                0, 2
            ),
        },
        index=frame.index,
    )


def prepare_taiwan(frame: pd.DataFrame, fairness: FairnessConfig) -> PreparedData:
    features = pd.concat(
        [frame[["LIMIT_BAL", *PAY_STATUS, *BILL, *PAY_AMT]], taiwan_engineered(frame)], axis=1
    ).astype(float)
    monotone = {c: 0 for c in features.columns}
    monotone.update(
        {
            "LIMIT_BAL": -1,
            "max_delay_6m": 1,
            "months_delayed_6m": 1,
            "utilisation": 1,
            "payment_ratio": -1,
            **{c: -1 for c in PAY_AMT},
        }
    )
    protected = pd.DataFrame(
        {
            "SEX": frame["SEX"].map({1: "Erkek", 2: "Kadın"}).fillna("Bilinmiyor"),
            "AGE_BAND": age_band(frame["AGE"], fairness),
            "EDUCATION": frame["EDUCATION"]
            .map({1: "Lisansüstü", 2: "Üniversite", 3: "Lise"})
            .fillna("Diğer/Bilinmiyor"),
            "MARRIAGE": frame["MARRIAGE"].map({1: "Evli", 2: "Bekar"}).fillna("Diğer/Bilinmiyor"),
        },
        index=frame.index,
    )
    return PreparedData(
        name="uci_taiwan",
        X=features,
        y=frame["default"].to_numpy(int),
        protected=protected,
        monotone=monotone,
        notes=[
            "SEX, AGE, EDUCATION ve MARRIAGE model girdisi değildir; yalnızca adillik analizinde kullanılır.",
            "Zaman ekseni yok: tabakalı 5 katlı CV + %20 tabakalı holdout.",
        ],
    )


# ------------------------------------------------------------------ German Credit
GERMAN_PROTECTED = {"personal_status_sex", "age", "foreign_worker"}
GERMAN_FEMALE = {"A92", "A95"}


def prepare_german(frame: pd.DataFrame, fairness: FairnessConfig) -> PreparedData:
    usable = frame.drop(columns=[*GERMAN_PROTECTED, "default"])
    categorical = [c for c in usable.columns if usable[c].dtype == object]
    features = pd.get_dummies(usable, columns=categorical, dtype=float).astype(float)
    monotone = {c: 0 for c in features.columns}
    monotone.update({"duration_months": 1})
    protected = pd.DataFrame(
        {
            "SEX": np.where(frame["personal_status_sex"].isin(GERMAN_FEMALE), "Kadın", "Erkek"),
            "AGE_BAND": age_band(frame["age"], fairness),
            "FOREIGN_WORKER": np.where(frame["foreign_worker"] == "A201", "Evet", "Hayır"),
        },
        index=frame.index,
    )
    return PreparedData(
        name="german_credit",
        X=features,
        y=frame["default"].to_numpy(int),
        protected=protected,
        monotone=monotone,
        notes=[
            "Kategorik alanlar tek-sıcak (one-hot) kodlandı; cinsiyet/medeni durum, yaş ve yabancı işçi alanları model dışı.",
            "1.000 kayıt: güven aralıkları geniştir, ikincil kontrol olarak yorumlanmalıdır.",
        ],
    )


PREPARERS = {"uci_taiwan": prepare_taiwan, "german_credit": prepare_german}


def prepare(name: str, frame: pd.DataFrame, fairness: FairnessConfig) -> PreparedData:
    return PREPARERS[name](frame, fairness)
