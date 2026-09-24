"""Population stability (PSI / CSI) drift monitoring.

Reference bins and shares are stored in the PD model metadata at training
time; live distributions come from the feature snapshots of recent engine
decisions. PSI < 0.10 stable, 0.10–0.25 moderate, > 0.25 significant shift
(alert). ``evidently`` can replace this as an optional extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Decision
from app.decisioning.models import get_models

MODERATE, SIGNIFICANT = 0.10, 0.25
_EPS = 1e-4


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, float), _EPS, None)
    a = np.clip(np.asarray(actual, float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def shares(values: list[float], edges: list[float]) -> list[float]:
    if not values:
        return [0.0] * (len(edges) - 1)
    clipped = np.clip(np.asarray(values, float), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return (counts / max(counts.sum(), 1)).tolist()


def status_for(value: float) -> str:
    return "ANLAMLI_KAYMA" if value > SIGNIFICANT else "ORTA" if value > MODERATE else "STABIL"


def compute_drift(
    snapshots: list[dict[str, Any]], reference: dict[str, dict[str, list[float]]]
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name, ref in reference.items():
        values = [float(s[name]) for s in snapshots if s.get(name) is not None]
        edges = ref["edges"]
        if len(edges) < 2 or not values:
            continue
        value = psi(ref["shares"], shares(values, edges))
        features[name] = {"psi": round(value, 4), "status": status_for(value), "n": len(values)}
    alerts = [
        f"{k}: PSI {v['psi']:.3f}" for k, v in features.items() if v["status"] == "ANLAMLI_KAYMA"
    ]
    return {
        "observations": len(snapshots),
        "features": features,
        "alerts": alerts,
        "max_psi": max((v["psi"] for v in features.values()), default=0.0),
    }


def compute_drift_report(session: Session, limit: int = 1000) -> dict[str, Any]:
    snapshots = [
        s
        for (s,) in session.execute(
            select(Decision.feature_snapshot)
            .where(Decision.kind == "engine")
            .order_by(Decision.created_at.desc())
            .limit(limit)
        ).all()
    ]
    reference = get_models().pd_model.meta.get("reference_distribution", {})
    report = compute_drift(snapshots, reference)
    report["model"] = get_models().pd_model.version
    return report
