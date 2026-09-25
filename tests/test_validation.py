"""Lane A validation on real public data: runner, report, datasets, evidence."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from app.core.rules import load_validation
from app.validation.datasets import FIXTURE_PATH, load_frame, prepare
from app.validation.lane_a import run_lane_a, select_champion
from app.validation.report import dataset_section


# ------------------------------------------------------------------ champion rule
def _test(diff: float, p: float) -> dict:
    return {"auc_diff": diff, "p_value": p}


def test_select_champion_needs_significance_and_materiality():
    holdout = {"lightgbm": {}, "logistic": {}, "scorecard": {}}
    order = ["scorecard", "logistic", "lightgbm"]
    kwargs = {"alpha": 0.05, "min_gain": 0.005, "simplicity_order": order}
    tests = {
        "logistic_vs_scorecard": _test(0.001, 0.6),
        "lightgbm_vs_scorecard": _test(0.02, 0.001),
    }
    assert select_champion(holdout, tests, **kwargs)["model"] == "lightgbm"
    tests["lightgbm_vs_scorecard"] = _test(0.003, 0.001)  # significant but immaterial
    assert select_champion(holdout, tests, **kwargs)["model"] == "scorecard"
    tests["lightgbm_vs_scorecard"] = _test(0.02, 0.2)  # material but not significant
    assert select_champion(holdout, tests, **kwargs)["model"] == "scorecard"
    flipped = {"scorecard_vs_logistic": _test(-0.01, 0.01), "logistic_vs_lightgbm": _test(0.0, 1)}
    assert select_champion(holdout, flipped, **kwargs)["model"] == "logistic"


# ------------------------------------------------------------------ runner
@pytest.fixture(scope="module")
def fixture_run():
    cfg = load_validation().model_copy(update={"cv_folds": 3, "bootstrap_iterations": 60})
    data = prepare("uci_taiwan", load_frame(FIXTURE_PATH), cfg.fairness)
    return run_lane_a(data, cfg)


def test_lane_a_on_fixture_produces_complete_metrics(fixture_run):
    m = fixture_run.metrics
    assert set(m["cv"]) == {"lightgbm", "logistic", "scorecard"}
    assert all(len(v["fold_auc"]) == 3 for v in m["cv"].values())
    for model, metrics in m["holdout"].items():
        assert 0.6 < metrics["auc"] < 0.9, model
        assert metrics["auc_ci"][0] < metrics["auc"] < metrics["auc_ci"][1]
    assert set(m["delong"]) == {
        "lightgbm_vs_logistic",
        "lightgbm_vs_scorecard",
        "logistic_vs_scorecard",
    }
    assert m["champion"]["model"] in m["holdout"] and m["champion"]["steps"]
    # Selection runs on the pooled out-of-fold predictions; the hold-out only confirms.
    assert m["champion"]["basis"] == "out_of_fold"
    assert set(m["delong_oof"]) == set(m["delong"])
    assert all(0.5 < v["oof_auc"] < 1 for v in m["cv"].values())
    spreads = [s["min_air_spread"] for s in m["fairness"]["by_model"]["lightgbm"].values()]
    assert all(s is None or s["min"] <= s["median"] <= s["max"] for s in spreads)
    assert m["fairness"]["tie_break"]["spread_seeds"] == load_validation().fairness.tie_break_seeds
    assert "lightgbm_uncalibrated" in m["calibration"]
    assert m["fairness"]["approval_rate"] == 0.7
    assert set(m["fairness"]["by_model"]["logistic"]) == {
        "SEX",
        "AGE_BAND",
        "EDUCATION",
        "MARRIAGE",
    }
    assert {r["approval_rate"] for r in m["lda"]["rows"]} == {0.7}
    assert m["shap_top_features"][0]["mean_abs_shap"] > 0
    # protected attributes are never model inputs
    assert not {"SEX", "AGE", "EDUCATION", "MARRIAGE"} & set(m["design"]["features"])
    json.dumps(m)  # serialisable


def test_report_renders_every_section(fixture_run):
    text = "\n".join(dataset_section("uci_taiwan", fixture_run.metrics, "img/validation"))
    for heading in (
        "Cross-validation",
        "Hold-out discrimination",
        "DeLong",
        "Champion",
        "Calibration",
        "Fairness at the same approval rate",
        "Less discriminatory alternatives",
        "SHAP",
    ):
        assert heading in text, heading


def test_german_preparation_excludes_protected_fields():
    cfg = load_validation()
    frame = pd.DataFrame(
        {
            "checking_status": ["A11", "A12", "A14", "A11"],
            "duration_months": [6, 48, 12, 24],
            "credit_amount": [1169, 5951, 2096, 7882],
            "personal_status_sex": ["A93", "A92", "A95", "A91"],
            "age": [67, 22, 49, 45],
            "foreign_worker": ["A201", "A202", "A201", "A201"],
            "default": [0, 1, 0, 1],
        }
    )
    data = prepare("german_credit", frame, cfg.fairness)
    assert "age" not in data.X and not any(c.startswith("personal_status") for c in data.X)
    assert list(data.protected.SEX) == ["Erkek", "Kadın", "Kadın", "Erkek"]
    assert data.monotone["duration_months"] == 1


def test_fetcher_normalises_german_archive_and_reports_kaggle_skip(monkeypatch):
    from scripts import fetch_public_credit_data as fetcher

    row = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 2\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("german.data", row * 3)
    frame = fetcher.normalise_german(buffer.getvalue())
    assert frame["default"].tolist() == [1, 1, 1] and "class" not in frame
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert all("atlandı" in v for v in fetcher.kaggle_status().values())


def test_fetcher_rejects_checksum_mismatch(monkeypatch, tmp_path):
    from scripts import fetch_public_credit_data as fetcher

    monkeypatch.setattr(fetcher, "EXTERNAL", tmp_path)
    monkeypatch.setattr(fetcher, "download", lambda url: b"not the archive")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetcher.fetch("german_credit", force=True)


@pytest.mark.external_data
def test_full_taiwan_data_matches_pinned_checksum():
    from app.validation.datasets import available, dataset_path

    if not available("uci_taiwan"):
        pytest.skip("full public data not downloaded")
    import hashlib

    digest = hashlib.sha256(dataset_path("uci_taiwan").read_bytes()).hexdigest()
    assert digest == "e28803eec99215182faffbced63edd940d5034ed240443084ae72d254f147f89"


# ------------------------------------------------------------------ governance evidence
def test_committed_evidence_and_promotion_rule(monkeypatch):
    from app.validation import evidence

    assert "uci_taiwan" in evidence.available_sets()
    metrics = evidence.load_metrics("uci_taiwan")
    assert metrics["source"] == "full" and metrics["design"]["rows"] == 30_000
    flipped = evidence.pair_test(metrics, "logistic", "lightgbm")
    direct = evidence.pair_test(metrics, "lightgbm", "logistic")
    assert flipped["auc_diff"] == -direct["auc_diff"]
    assert flipped["diff_ci"] == [-direct["diff_ci"][1], -direct["diff_ci"][0]]
    assert evidence.pair_test(metrics, "lightgbm", "unknown") is None
    decision = evidence.promotion_evidence("lightgbm_monotone", "logistic_regression")
    assert decision["allowed"] is False and "zayıf" in decision["reason"]
    reverse = evidence.promotion_evidence("logistic_regression", "lightgbm_monotone")
    assert reverse["allowed"] is True
    assert evidence.promotion_evidence("x", "y")["allowed"] is False
    monkeypatch.setattr(evidence, "load_metrics", lambda name: None)
    assert evidence.family_evidence("lightgbm", "logistic")["available"] is False
    assert (
        evidence.promotion_evidence("lightgbm_monotone", "logistic_regression")["allowed"] is False
    )


def test_promotion_gate_checks_every_dataset_and_refuses_unknown_families():
    from app.validation import evidence

    unknown = evidence.promotion_evidence("lightgbm_monotone", "ebm")
    assert unknown["allowed"] is False and "ebm" in unknown["reason"]
    assert unknown["evidence_scope"] == evidence.EVIDENCE_SCOPE
    # Scorecard is not worse on German Credit but significantly worse on Taiwan: refused.
    decision = evidence.promotion_evidence("lightgbm_monotone", "optbinning_woe_logistic")
    assert {d["dataset"] for d in decision["datasets"]} == set(evidence.available_sets())
    worse = {d["dataset"]: d["significantly_worse"] for d in decision["datasets"]}
    assert worse == {"uci_taiwan": True, "german_credit": False}
    assert decision["allowed"] is False and decision["families"]["challenger"] == "scorecard"


def test_committed_evidence_uses_out_of_fold_selection_and_reports_untestable_groups():
    from app.validation import evidence

    for name in evidence.available_sets():
        metrics = evidence.load_metrics(name)
        assert metrics["champion"]["basis"] == "out_of_fold", name
        assert set(metrics["delong_oof"]) == set(metrics["delong"]), name
    german = evidence.load_metrics("german_credit")
    foreign = german["fairness"]["by_model"]["scorecard"]["FOREIGN_WORKER"]
    assert foreign["testable"] is False and foreign["min_air"] is None
    assert foreign["passes_four_fifths"] is None and foreign["min_air_spread"] is None
    rec = german["lda"]["recommendation"]
    assert rec["kind"] == "model_family" and rec["recommended"] == "lightgbm"
    assert rec["min_air_to"] < load_validation().fairness.air_threshold  # still an open finding
