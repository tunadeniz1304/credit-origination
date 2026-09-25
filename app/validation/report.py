"""Markdown rendering of the lane A / lane B validation metrics."""

from __future__ import annotations

from typing import Any

MODEL_NAMES = {
    "lightgbm": "Monotone LightGBM + isotonic",
    "lightgbm_uncalibrated": "Monotone LightGBM (uncalibrated)",
    "logistic": "Logistic regression",
    "scorecard": "WoE scorecard (optbinning)",
}
VERDICT_EN = {
    "selected": "selected",
    "not_significant": "not significant — simpler model kept",
    "immaterial": "significant but below the materiality threshold — simpler model kept",
    "worse": "significantly worse — simpler model kept",
}
LDA_KIND_EN = {
    "baseline": "Unconstrained logistic regression",
    "model_family": "Other trained family: {family}",
    "proxy_removal": "Proxy-weakened logistic regression",
    "exponentiated_gradient": "ExponentiatedGradient, demographic parity (ε={epsilon})",
    "group_threshold": "Group-specific thresholds, demographic parity (ThresholdOptimizer-style)",
    "threshold_optimizer_reference": "fairlearn ThresholdOptimizer (own operating point)",
}
DESIGN_NOTES = {
    "uci_taiwan": [
        "SEX, AGE, EDUCATION and MARRIAGE are not model inputs; they are used only for fairness.",
        "No time axis: stratified 5-fold CV on the training part plus a 20 % stratified hold-out.",
        "`PAY_0` is the September 2005 repayment status, the month before the target month "
        "(see DATA.md) — no same-month leakage.",
    ],
    "german_credit": [
        "Categorical fields one-hot encoded; personal status/sex, age and foreign-worker flag excluded.",
        "Only 1,000 rows: confidence intervals are wide; treat as a secondary check.",
    ],
}
DATASET_NAMES = {
    "uci_taiwan": "UCI Default of Credit Card Clients (Taiwan, 30,000 rows)",
    "german_credit": "Statlog German Credit (1,000 rows)",
}


def _ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _num(value: float | None, digits: int = 3) -> str:
    """Fairness figures: ``None`` means *not testable* and is shown as n/a."""
    return "n/a" if value is None else f"{value:.{digits}f}"


AIR_VERDICTS_EN = {
    "passes": "passes",
    "fails": "fails",
    "inconclusive": "indicative, not statistically established",
}


def _air_with_spread(stats: dict[str, Any]) -> str:
    spread = stats.get("min_air_spread")
    if stats["min_air"] is None:
        return "n/a"
    text = _num(stats["min_air"])
    if spread:
        text += f" (seeds {spread['min']:.3f}–{spread['max']:.3f}, median {spread['median']:.3f})"
    boot = stats.get("min_air_ci")
    if boot:
        text += f"; 95% CI [{boot['ci'][0]:.3f}, {boot['ci'][1]:.3f}] {AIR_VERDICTS_EN[boot['verdict']]}"
    return text


def _opt_ci(ci: list[float] | None, signed: bool = False) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if signed else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _lda_name(row: dict[str, Any], champion: str) -> str:
    if row["kind"] == "champion":
        return f"Champion: {MODEL_NAMES[champion]}"
    family = MODEL_NAMES.get(row.get("family", ""), row.get("family"))
    return LDA_KIND_EN[row["kind"]].format(epsilon=row.get("epsilon"), family=family)


def _recommendation(rec: dict[str, Any], rows: list[dict[str, Any]], champion: str) -> str:
    if rec.get("testable") is False:
        return "not testable — only one group of the attribute meets the minimum size."
    basis = (
        "out-of-fold and hold-out" if rec.get("basis") == "out_of_fold_and_holdout" else "hold-out"
    )
    if not rec.get("recommended"):
        text = (
            f"keep the champion — no alternative raises the minimum AIR within the allowed AUC loss "
            f"({rec.get('max_auc_loss', 0):.3f}, {basis})."
        )
        rejected = [
            f"{_lda_name(next(r for r in rows if r['model'] == x['model']), champion)} "
            f"(min AIR {x['min_air']:.3f}"
            + (
                f", AIR gain 95% CI {_opt_ci(x['air_gain_ci'], signed=True)}"
                if x.get("air_gain_ci")
                else ""
            )
            + f", AUC loss {_loss(x)})"
            for x in rec.get("rejected", [])
        ]
        if rejected:
            text += " Fairer but outside the limit: " + "; ".join(rejected) + "."
        return text
    row = next(r for r in rows if r["model"] == rec["recommended"])
    gain = ""
    if rec.get("air_gain_ci"):
        gain = f", AIR gain 95% CI {_opt_ci(rec['air_gain_ci'], signed=True)}" + (
            ""
            if rec.get("air_gain_established")
            else " (contains 0: gain not statistically established)"
        )
    return (
        f"{_lda_name(row, champion)} — min AIR {rec['min_air_from']:.3f} → {rec['min_air_to']:.3f}"
        f"{gain} for an AUC loss of {_loss(rec)} (limit {rec['max_auc_loss']:.3f} on {basis}); the "
        "model risk committee decides whether to adopt it."
    )


def _loss(x: dict[str, Any]) -> str:
    """AUC loss text: hold-out, plus out-of-fold when the table carries it."""
    text = f"{x['auc_loss']:.4f} hold-out"
    if x.get("oof_auc_loss") is not None:
        text += f", {x['oof_auc_loss']:.4f} out-of-fold"
    return text


def dataset_section(name: str, m: dict[str, Any], image_dir: str) -> list[str]:
    design = m["design"]
    out = [
        f"## {DATASET_NAMES.get(name, name)}",
        "",
        f"Rows {design['rows']:,} · default rate {design['default_rate']:.2%} · train {design['train_rows']:,} "
        f"({design['cv_folds']}-fold stratified CV) · stratified hold-out {design['holdout_rows']:,} "
        f"({design['holdout_fraction']:.0%}) · seed {design['seed']} · "
        f"{len(design['features'])} own features.",
        "",
    ]
    out += [f"- {note}" for note in DESIGN_NOTES.get(name, design["notes"])] + [""]

    out += ["### Cross-validation (training part) — champion selection", ""]
    out += _table(
        ["Model", "Mean fold AUC", "Std", "Pooled out-of-fold AUC", "Fold AUCs"],
        [
            [
                MODEL_NAMES[k],
                f"{v['mean_auc']:.4f}",
                f"{v['std_auc']:.4f}",
                f"{v['oof_auc']:.4f}" if "oof_auc" in v else "—",
                ", ".join(map(str, v["fold_auc"])),
            ]
            for k, v in m["cv"].items()
        ],
    )
    if m.get("delong_oof"):
        out += [
            "",
            "Paired DeLong tests on the pooled out-of-fold scores (the selection evidence):",
            "",
        ]
        out += _table(
            ["Comparison", "ΔAUC", "95% CI of Δ", "z", "p-value"],
            [
                [
                    " vs ".join(MODEL_NAMES[p] for p in key.split("_vs_")),
                    f"{t['auc_diff']:+.4f}",
                    _ci(t["diff_ci"]),
                    f"{t['z']:.2f}",
                    f"{t['p_value']:.3g}",
                ]
                for key, t in m["delong_oof"].items()
            ],
        )
    out += ["", "### Hold-out discrimination", ""]
    out += _table(
        [
            "Model",
            "AUC",
            "95% CI (bootstrap)",
            "95% CI (DeLong)",
            "Gini",
            "KS",
            "Brier",
            "Log loss",
        ],
        [
            [
                MODEL_NAMES[k],
                f"{v['auc']:.4f}",
                _ci(v["auc_ci"]),
                _ci(v["auc_ci_delong"]),
                f"{v['gini']:.4f}",
                f"{v['ks']:.4f}",
                f"{v['brier']:.4f}",
                f"{v['log_loss']:.4f}",
            ]
            for k, v in m["holdout"].items()
        ],
    )
    out += [
        "",
        "DeLong tests on the hold-out (two-sided; confirmation only, not used for the choice):",
        "",
    ]
    out += _table(
        ["Comparison", "ΔAUC", "95% CI of Δ", "z", "p-value"],
        [
            [
                " vs ".join(MODEL_NAMES[p] for p in key.split("_vs_")),
                f"{t['auc_diff']:+.4f}",
                _ci(t["diff_ci"]),
                f"{t['z']:.2f}",
                f"{t['p_value']:.3g}",
            ]
            for key, t in m["delong"].items()
        ],
    )
    champion = m["champion"]
    oof = champion.get("basis") == "out_of_fold"
    out += [
        "",
        f"**Champion: {MODEL_NAMES[champion['model']]}.** Rule: a more complex model replaces a simpler "
        "one only when the DeLong test is significant *and* the AUC gain is material (thresholds in "
        "`rules/validation.yaml`). "
        + (
            "The choice is made on the pooled out-of-fold scores of the training part; the hold-out "
            "only confirms it, so the hold-out figures above are not selection-biased."
            if oof
            else "The choice was made on the hold-out."
        ),
        "",
    ]
    out += [
        f"- {MODEL_NAMES[c['challenger']]} vs {MODEL_NAMES[c['incumbent']]}: "
        f"ΔAUC {c['auc_diff']:+.4f}, DeLong p = {c['p_value']:.3g}"
        + (" (out-of-fold)" if oof else "")
        + f" → {VERDICT_EN[c['verdict']]}"
        + (
            f"; hold-out confirmation ΔAUC {c['holdout_auc_diff']:+.4f}, "
            f"p = {c['holdout_p_value']:.3g}"
            if c.get("holdout_auc_diff") is not None
            else ""
        )
        for c in champion["comparisons"]
    ] + [
        f"- Thresholds: α = {champion['alpha']}, minimum ΔAUC = {champion['min_auc_gain']}.",
        "",
    ]

    out += ["### Calibration (hold-out)", ""]
    out += _table(
        [
            "Model",
            "Mean PD",
            "Observed",
            "ECE",
            "Hosmer–Lemeshow χ² (p)",
            "Low-risk deciles: predicted → observed (ratio)",
        ],
        [
            [
                MODEL_NAMES[k],
                _pct(c["mean_predicted"]),
                _pct(c["observed_rate"]),
                f"{c['ece']:.4f}",
                f"{c['hosmer_lemeshow']['statistic']:.1f} ({c['hosmer_lemeshow']['p_value']:.3g})",
                f"{_pct(c['low_risk']['predicted'])} → {_pct(c['low_risk']['observed'])} "
                f"({c['low_risk']['ratio_observed_to_predicted']})",
            ]
            for k, c in m["calibration"].items()
        ],
    )
    champ_cal = m["calibration"][champion["model"]]
    out += ["", f"Decile table of the champion ({MODEL_NAMES[champion['model']]}):", ""]
    out += _table(
        ["Decile", "n", "PD range", "Predicted", "Observed", "Obs / pred"],
        [
            [
                r["decile"],
                r["n"],
                f"{r['pd_min']:.3f}–{r['pd_max']:.3f}",
                _pct(r["predicted"]),
                _pct(r["observed"]),
                r["ratio_observed_to_predicted"],
            ]
            for r in champ_cal["deciles"]
        ],
    )
    out += [
        "",
        f"![Calibration]({image_dir}/{name}_calibration.png) ![ROC]({image_dir}/{name}_roc.png)",
        "",
    ]

    fairness = m["fairness"]
    tie = fairness.get("tie_break", {})
    out += [
        f"### Fairness at the same approval rate ({fairness['approval_rate']:.0%})",
        "",
        "Every model approves the same share of the hold-out (lowest PDs first). AIR = group approval "
        "rate / highest group approval rate; TPR = approval rate of good payers, FPR = approval rate of "
        "defaulters (equalised odds). Groups smaller than "
        f"{fairness.get('min_group_size', 0)} hold-out rows are excluded from AIR (unstable rates); "
        "an attribute left with a single group is **not testable** (n/a), not a perfect 1.000.",
        "",
        "Applicants with the same PD at the cut-off (isotonic calibration yields few distinct PDs) are "
        f"approved in a seeded random order, never by row order. Min AIR is shown for seed "
        f"{tie.get('seed', '—')} followed by (min–max, median) over {tie.get('spread_seeds', '—')} "
        "tie-break seeds; a conclusion that flips inside that range rests on an arbitrary choice. "
        "The seed spread covers **only** the tie-breaking, not the sampling error of the hold-out.",
        "",
    ]
    boot = fairness.get("bootstrap")
    if boot:
        out += [
            f"**Sampling error.** The {boot['confidence']:.0%} CI of the minimum AIR is a percentile "
            f"bootstrap ({boot['iterations']} resamples, seed {boot['seed']}). {boot['note']} "
            "When the interval contains the four-fifths threshold the pass/fail reading is "
            "**indicative, not statistically established**.",
            "",
        ]
    by_model = fairness["by_model"]
    attrs = list(next(iter(by_model.values())))
    out += _table(
        ["Model", *[f"min AIR {a}" for a in attrs], *[f"TPR gap {a}" for a in attrs]],
        [
            [
                MODEL_NAMES[k],
                *[_air_with_spread(v[a]) for a in attrs],
                *[_num(v[a]["tpr_gap"]) for a in attrs],
            ]
            for k, v in by_model.items()
        ],
    )
    out += ["", f"Group detail for the champion ({MODEL_NAMES[champion['model']]}):", ""]
    rows = []
    for attr, stats in by_model[champion["model"]].items():
        for group, rate in stats["selection_rate"].items():
            rows.append(
                [
                    attr,
                    group,
                    stats["group_size"][group],
                    _pct(rate),
                    _num(stats["air"][group]),
                    _pct(stats["tpr_good_approved"][group]),
                    _pct(stats["fpr_bad_approved"][group]),
                ]
            )
    out += _table(["Attribute", "Group", "n", "Approval", "AIR", "TPR (goods)", "FPR (bads)"], rows)

    lda = m["lda"]
    out += [
        "",
        f"### Less discriminatory alternatives ({lda['attribute']}, all rows at {lda['approval_rate']:.0%} approval)",
        "",
    ]
    out += _table(
        [
            "Alternative",
            "Hold-out AUC",
            "Out-of-fold AUC",
            "Approval",
            "Bad rate of approved",
            "Min AIR",
            "Min AIR 95% CI",
            "AIR gain vs champion 95% CI",
            "TPR gap",
            "FPR gap",
        ],
        [
            [
                _lda_name(r, champion["model"]),
                f"{r['auc']:.4f}",
                f"{r['oof_auc']:.4f}" if r.get("oof_auc") is not None else "—",
                _pct(r["approval_rate"]),
                _pct(r["bad_rate_approved"]),
                _num(r["min_air"]),
                _opt_ci(r.get("min_air_ci")),
                _opt_ci(r.get("min_air_gain_ci"), signed=True),
                _num(r["tpr_gap"]),
                _num(r["fpr_gap"]),
            ]
            for r in lda["rows"]
        ],
    )
    dropped: list[str] = next(
        (r.get("dropped_features") for r in lda["rows"] if r["kind"] == "proxy_removal"), []
    )
    out += [
        "",
        f"- Proxy strength (single-feature AUC for {lda['attribute']}): "
        + ", ".join(f"`{k}` {v:.3f}" for k, v in list(lda["proxy_strength"].items())[:5])
        + f". Dropped in the proxy-weakened model: {', '.join(f'`{d}`' for d in dropped) or '—'}.",
        "- Recommendation: "
        + _recommendation(lda["recommendation"], lda["rows"], champion["model"]),
        "- Group-specific thresholds use the protected attribute at decision time; they are shown only "
        "to quantify the trade-off and are legally problematic (direct discrimination).",
    ]
    for ref in lda.get("reference_rows", []):
        out.append(
            f"- Reference, not comparable (own operating point): {_lda_name(ref, champion['model'])} — approval "
            f"{_pct(ref['approval_rate'])}, min AIR {_num(ref['min_air'])}."
        )
    out += ["", "### Largest SHAP contributions (LightGBM, hold-out sample)", ""]
    out += _table(
        ["Feature", "mean |SHAP|"],
        [[f"`{r['feature']}`", r["mean_abs_shap"]] for r in m["shap_top_features"]],
    )
    out.append("")
    return out


def _policy_reading(policy: dict[str, Any]) -> list[str]:
    """Provenance of the policy table and the two comparisons behind the v2 cut-offs."""
    out: list[str] = []
    digests = policy.get("model_sha256") or {}
    if digests:
        out += [
            "Scored model files (SHA-256, LF line endings): "
            + ", ".join(f"`{path}` `{digest[:16]}…`" for path, digest in digests.items())
            + ". Thresholds read from `rules/policy_v2.yaml`.",
            "",
        ]
    rows = {r["policy"]: r for r in policy["rows"]}
    v2, same_model = rows.get("policy_v2"), rows.get("policy_v1 thresholds")
    if not v2 or not same_model:
        return out
    before, after = same_model["bad_rate_auto_approved"], v2["bad_rate_auto_approved"]
    text = (
        f"**Risk appetite.** On the same v2 model, the v2 cut-offs raise the auto-approved share "
        f"from {_pct(same_model['share_auto_approve'])} to {_pct(v2['share_auto_approve'])} and the "
        f"bad rate of the auto-approved book from {_pct(before)} to {_pct(after)} "
        f"({after / before - 1:+.0%} relative): the v2 cut-offs **loosen** the risk appetite; they "
        "do not re-express the v1 appetite on the new scale."
    )
    retired = rows.get("policy_v1")
    if retired and retired.get("mean_pd_auto_approved"):
        under = retired["bad_rate_auto_approved"] / retired["mean_pd_auto_approved"]
        text += (
            f" The comparison with the retired v1 model ({_pct(retired['bad_rate_auto_approved'])} "
            f"→ {_pct(after)}) is not a like-for-like baseline: on this population that model "
            f"under-predicts the auto-approved book about {under:.1f}× (mean PD "
            f"{_pct(retired['mean_pd_auto_approved'])}, realised {_pct(retired['bad_rate_auto_approved'])})."
        )
    text += (
        " Adopting the v2 cut-offs is a risk-appetite change that needs credit committee sign-off."
    )
    return [*out, text, ""]


def lane_b_section(lane: dict[str, Any]) -> list[str]:
    """Lane B: mapping, behaviour sub-score, generator anchoring, PD level check."""
    anchors = lane["anchors"]
    check = lane["generator_check"]
    calib = lane.get("production_calibration", {})
    behaviour = lane.get("behaviour_score", {})
    out = [
        "## Lane B — anchoring the production model to a real proxy curve (not a validation)",
        "",
        "The production model runs on Turkey-specific features that no public set contains, so it "
        "cannot be validated end-to-end on public data. Its bureau-behaviour inputs are mapped onto "
        "the Taiwan variables (`app/decisioning/public_mapping.py`, table in [`DATA.md`](DATA.md)), "
        "a behaviour sub-score is learnt on real defaults, and the synthetic generator is anchored "
        "to the real default curve. **The PD level is imposed from a real proxy curve (anchoring), "
        "not validated:** the anchor is next-month credit-card default, used as the level of a "
        "12-month 90+DPD personal-loan PD. The synthetic outcomes are then drawn from that same "
        "curve, so the checks below are internal consistency checks — they show the level was "
        "imposed correctly, not that it is right. The only real-data evidence is lane A "
        "(methodology) and the behaviour sub-score's real hold-out AUC.",
        "",
        f"- Real anchor source: {anchors['source']} ({anchors['rows']:,} rows, default rate "
        f"{anchors['overall_default_rate']:.2%}).",
        f"- `bureau_behavior_score` ({behaviour.get('version', '—')}): monotone LightGBM on delay "
        f"months, delinquent months and utilisation; real hold-out AUC "
        f"{behaviour.get('holdout_auc', float('nan')):.3f} (n = {behaviour.get('holdout_rows', 0):,}).",
        "",
        f"### Generator check — default rate per delinquency band (tolerance ±{check['tolerance_abs']:.2f})",
        "",
    ]
    out += _table(
        [
            "Max delay (months)",
            "Real default rate",
            "Synthetic default rate",
            "|gap|",
            "Within",
            "Real share",
            "Synthetic share",
        ],
        [
            [
                f"{c['band']}+" if c is check["bands"][-1] else c["band"],
                _pct(c["real_default_rate"]),
                _pct(c["synthetic_default_rate"]),
                f"{c['abs_gap']:.4f}",
                "yes" if c["within_tolerance"] else "**no**",
                _pct(c["real_share"]),
                _pct(c["synthetic_share"]),
            ]
            for c in check["bands"]
        ],
    )
    out += ["", f"- {check['note']}", ""]
    if calib:
        low = calib["low_risk"]
        out += [
            f"### Production PD ({calib['model']}) — level and low-risk calibration "
            "(anchored synthetic population)",
            "",
            f"Synthetic time-based test set, n = {calib['test_rows']:,}: ECE {calib['ece']:.4f}, "
            f"Hosmer–Lemeshow χ² {calib['hosmer_lemeshow']['statistic']:.1f} "
            f"(p = {calib['hosmer_lemeshow']['p_value']:.3g}). Lowest {low['deciles']} deciles: "
            f"predicted {_pct(low['predicted'])}, observed {_pct(low['observed'])} "
            f"(ratio {low['ratio_observed_to_predicted']}; tolerance ±{calib['tolerance_ratio']:.0%} — "
            f"{'met' if calib['low_risk_within_tolerance'] else '**not met**'} in aggregate, "
            f"{'met' if calib.get('low_risk_deciles_within_tolerance') else '**not met**'} per decile).",
            "",
        ]
        out += [
            "Deciles are formed on average ranks, so applicants with the same PD always share a "
            "decile (isotonic calibration produces large tie blocks; decile sizes therefore differ).",
            "",
        ]
        out += _table(
            ["Decile", "n", "Predicted", "Observed", "Obs / pred"],
            [
                [
                    d["decile"],
                    d["n"],
                    _pct(d["predicted"]),
                    _pct(d["observed"]),
                    d["ratio_observed_to_predicted"],
                ]
                for d in calib["deciles"]
            ],
        )
        out += ["", "Mean predicted PD per delinquency band vs the real default rate:", ""]
        out += _table(
            ["Band", "Real default rate", "Mean predicted PD", "n (test)"],
            [
                [b["band"], _pct(b["real_default_rate"]), _pct(b["mean_predicted_pd"]), b["n"]]
                for b in calib["per_band_level"]
            ],
        )
        out.append("")
    policy = lane.get("policy_cutoffs")
    if policy:
        out += [
            "### Policy cut-off table (anchored synthetic population)",
            "",
            f"{policy['population']}, n = {policy['test_rows']:,}, observed default rate "
            f"{_pct(policy['observed_default_rate'])}. {policy['note']}",
            "",
        ]
        out += _table(
            [
                "Policy",
                "Model",
                "Auto-approve PD ≤",
                "Auto-decline PD ≥",
                "Auto-approved",
                "Referred (grey)",
                "Auto-declined",
                "Bad rate of auto-approved",
            ],
            [
                [
                    r["policy"],
                    r["model"],
                    _pct(r["auto_approve_max_pd"]),
                    _pct(r["auto_decline_min_pd"]),
                    _pct(r["share_auto_approve"]),
                    _pct(r["share_referred"]),
                    _pct(r["share_auto_decline"]),
                    _pct(r["bad_rate_auto_approved"]),
                ]
                for r in policy["rows"]
            ],
        )
        out.append("")
        out += _policy_reading(policy)
    out += [
        "**Reading this honestly.** The anchoring transfers the *shape* of real credit risk "
        "(how default rises with arrears) into the synthetic population and imposes a PD level "
        "from a proxy: the Taiwan target is next-month default on credit cards, not 90+ DPD within "
        "12 months on personal loans, and Taiwanese card holders are not Turkish loan applicants. "
        "Whether the old model under- or over-stated risk cannot be decided from this data. The "
        "anchored model is a methodologically sound starting point, not a validated Turkish PD "
        "model; a real bank portfolio is needed for re-training and independent validation.",
        "",
    ]
    return out
