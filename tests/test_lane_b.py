"""Lane B: public→platform mapping, behaviour sub-score, anchored generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.rules import load_validation
from app.decisioning import public_mapping as pm
from app.decisioning.training import SEED, anchor_to_real_curve, generate_dataset
from app.validation.datasets import FIXTURE_PATH, load_frame


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return generate_dataset(20_000, SEED)


def test_map_public_ranges_and_semantics():
    public = pm.map_public(load_frame(FIXTURE_PATH))
    assert set(pm.BEHAVIOUR_FEATURES) <= set(public.columns)
    assert public["delay_months"].between(0, pm.MAX_DELAY_MONTHS).all()
    assert public["delinquent_months"].between(0, 6).all()
    assert public["utilisation"].between(0, pm.MAX_UTILISATION).all()
    # Arrears make default more likely in the real data (the curve we anchor to).
    curve = pm.default_curve(public["delay_months"], public["default"], [0, 1, 2, 3])
    assert curve["0"]["default_rate"] < curve["2"]["default_rate"] < curve["3"]["default_rate"]
    assert sum(c["share"] for c in curve.values()) == pytest.approx(1.0, abs=1e-3)


def test_map_platform_converts_days_to_months_and_caps():
    mapped = pm.map_platform(
        {"max_dpd_24m": 45, "delinquency_count_24m": 9, "bureau_utilisation": 2.0}
    ).iloc[0]
    assert mapped["delay_months"] == 2 and mapped["delinquent_months"] == 6
    assert mapped["utilisation"] == pm.MAX_UTILISATION
    assert pm.delinquency_band([0, 1, 2, 5, 8], [0, 1, 2, 3]).tolist() == [0, 1, 2, 3, 3]


def test_behaviour_score_is_monotone_in_arrears_and_utilisation():
    score = pm.load_behaviour_score()
    grid = pd.DataFrame(
        {
            "max_dpd_24m": [0, 30, 60, 90, 0, 0],
            "delinquency_count_24m": [0, 1, 2, 3, 0, 0],
            "bureau_utilisation": [0.3, 0.3, 0.3, 0.3, 0.1, 0.95],
        }
    )
    values = score.score_platform(grid)
    assert np.all(np.diff(values[:4]) >= 0) and values[3] > values[0]
    assert values[5] >= values[4]
    assert score.meta["metrics"]["holdout_auc"] > 0.7


def test_snapshot_carries_behaviour_score_only_with_a_bureau_record():
    assert pm.behaviour_score_for({"bureau_hit": 0}) is None
    value = pm.behaviour_score_for(
        {
            "bureau_hit": 1,
            "max_dpd_24m": 0,
            "delinquency_count_24m": 0,
            "bureau_utilisation": 0.2,
        }
    )
    assert 0 < value < 0.25


def test_generator_matches_real_default_curve_per_delinquency_band(synthetic):
    """Spec: synthetic vs real default rate per band within the configured tolerance."""
    cfg = load_validation().lane_b
    anchors = pm.lane_b_anchors()["delinquency_band_default"]
    hit = synthetic["bureau_hit"] == 1
    curve = pm.default_curve(
        np.ceil(synthetic.loc[hit, "max_dpd_24m"] / pm.DAYS_PER_MONTH),
        synthetic.loc[hit, "target"],
        cfg.delinquency_bands,
    )
    for band, real in anchors.items():
        gap = abs(curve[band]["default_rate"] - real["default_rate"])
        assert gap <= cfg.band_tolerance_abs, (band, curve[band], real)


def test_generator_proxies_correlate_with_protected_attributes(synthetic):
    assert synthetic["age"].corr(synthetic["employment_months"]) > 0.2
    metro = synthetic["province"].isin(["İstanbul", "Ankara", "İzmir"])
    assert synthetic.loc[metro, "log_income"].mean() > synthetic.loc[~metro, "log_income"].mean()
    assert (synthetic["employment_months"] <= (synthetic["age"] - 18) * 12).all()


def test_anchor_shift_hits_target_and_thin_files_follow_clean_band():
    rng = np.random.default_rng(0)
    logit = rng.normal(-2, 1, 4000)
    max_dpd = np.repeat([0, 30, 60, 120], 1000)
    hit = np.ones(4000, dtype=int)
    hit[:200] = 0
    shifted = anchor_to_real_curve(logit, max_dpd, hit)
    anchors = pm.lane_b_anchors()["delinquency_band_default"]
    p = 1 / (1 + np.exp(-shifted))
    assert p[1000:2000].mean() == pytest.approx(anchors["1"]["default_rate"], abs=1e-4)
    assert p[200:1000].mean() == pytest.approx(anchors["0"]["default_rate"], abs=1e-4)
    thin_shift = shifted[:200] - logit[:200]
    assert np.allclose(thin_shift, (shifted[200:1000] - logit[200:1000])[0])


def test_committed_lane_b_evidence_is_within_tolerance():
    import json

    lane = json.loads(pm.LANE_B_PATH.read_text(encoding="utf-8"))
    assert lane["anchors"]["source"] == "uci_taiwan (full)"
    assert lane["generator_check"]["all_within_tolerance"]
    calib = lane["production_calibration"]
    assert calib["low_risk_within_tolerance"]
    # Per decile the lowest (tie-aware) decile is under-predicted: an open, documented finding.
    # The flag must follow the committed deciles, never be asserted into existence.
    tolerance = calib["tolerance_ratio"]
    low = calib["deciles"][: calib["low_risk"]["deciles"]]
    assert calib["low_risk_deciles_within_tolerance"] == all(
        abs(d["ratio_observed_to_predicted"] - 1) <= tolerance for d in low
    )
    assert sum(d["n"] for d in calib["deciles"]) == calib["test_rows"]
    for band in calib["per_band_level"][:2]:  # large bands: level matches the real rate
        assert abs(band["mean_predicted_pd"] - band["real_default_rate"]) < 0.05


def test_documented_policy_cutoffs_match_the_artifact():
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lane = json.loads(pm.LANE_B_PATH.read_text(encoding="utf-8"))
    rows = {row["policy"]: row for row in lane["policy_cutoffs"]["rows"]}
    v1, v1_on_v2, v2 = rows["policy_v1"], rows["policy_v1 thresholds"], rows["policy_v2"]

    def pct(value: float) -> str:
        return f"{value * 100:.1f} %"

    card = (root / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    table = {
        line.split("|")[1].strip(): [c.strip().strip("*") for c in line.split("|")[3:7]]
        for line in card.splitlines()
        if re.match(r"\| \**v[12]", line)
    }
    for label, row in (
        ("v1 (PD ≤ 5 % / ≥ 20 %)", v1),
        ("v1 cut-offs on the v2 model", v1_on_v2),
        ("**v2 (PD ≤ 8 % / ≥ 30 %)**", v2),
    ):
        assert table[label] == [
            pct(row["share_auto_approve"]),
            pct(row["share_referred"]),
            pct(row["share_auto_decline"]),
            pct(row["bad_rate_auto_approved"]),
        ], label
    assert v2["bad_rate_auto_approved"] <= v1["bad_rate_auto_approved"]
    # The same numbers, rounded, in the policy file comment and the final report.
    policy = (root / "rules" / "policy_v2.yaml").read_text(encoding="utf-8")
    report = (root / "docs" / "FINAL_REPORT_v2.md").read_text(encoding="utf-8")
    for text in (policy, report):
        for value in (v1["bad_rate_auto_approved"], v2["bad_rate_auto_approved"]):
            assert f"{value * 100:.1f}".replace(".", ",") in text or pct(value) in text


def test_train_behaviour_score_on_fixture(tmp_path):
    public = pm.map_public(load_frame(FIXTURE_PATH))
    meta = pm.train_behaviour_score(public, seed=1, holdout_fraction=0.3, out_dir=tmp_path)
    assert (tmp_path / f"{pm.BEHAVIOUR_VERSION}.txt").is_file()
    assert 0.6 < meta["metrics"]["holdout_auc"] < 0.9
    fresh = pm.load_behaviour_score(str(tmp_path))
    assert fresh.meta["version"] == pm.BEHAVIOUR_VERSION
    pm.load_behaviour_score.cache_clear()
