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


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _lda_name(row: dict[str, Any], champion: str) -> str:
    if row["kind"] == "champion":
        return f"Champion: {MODEL_NAMES[champion]}"
    return LDA_KIND_EN[row["kind"]].format(epsilon=row.get("epsilon"))


def _recommendation(rec: dict[str, Any], rows: list[dict[str, Any]], champion: str) -> str:
    if not rec.get("recommended"):
        return (
            f"keep the champion — no alternative raises the minimum AIR within the allowed AUC loss "
            f"({rec.get('max_auc_loss', 0):.3f})."
        )
    row = next(r for r in rows if r["model"] == rec["recommended"])
    return (
        f"{_lda_name(row, champion)} — min AIR {rec['min_air_from']:.3f} → {rec['min_air_to']:.3f} "
        f"for an AUC loss of {rec['auc_loss']:.4f} (limit {rec['max_auc_loss']:.3f}); the model risk "
        "committee decides whether to adopt it."
    )


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

    out += ["### Cross-validation (training part)", ""]
    out += _table(
        ["Model", "Mean AUC", "Std", "Fold AUCs"],
        [
            [
                MODEL_NAMES[k],
                f"{v['mean_auc']:.4f}",
                f"{v['std_auc']:.4f}",
                ", ".join(map(str, v["fold_auc"])),
            ]
            for k, v in m["cv"].items()
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
    out += ["", "DeLong tests on the hold-out (two-sided):", ""]
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
    out += [
        "",
        f"**Champion: {MODEL_NAMES[champion['model']]}.** Rule: a more complex model replaces a simpler "
        "one only when the DeLong test is significant *and* the AUC gain is material (thresholds in "
        "`rules/validation.yaml`).",
        "",
    ]
    out += [
        f"- {MODEL_NAMES[c['challenger']]} vs {MODEL_NAMES[c['incumbent']]}: "
        f"ΔAUC {c['auc_diff']:+.4f}, DeLong p = {c['p_value']:.3g} → {VERDICT_EN[c['verdict']]}"
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
    out += [
        f"### Fairness at the same approval rate ({fairness['approval_rate']:.0%})",
        "",
        "Every model approves the same share of the hold-out (lowest PDs first). AIR = group approval "
        "rate / highest group approval rate; TPR = approval rate of good payers, FPR = approval rate of "
        "defaulters (equalised odds). Groups smaller than "
        f"{fairness.get('min_group_size', 0)} hold-out rows are excluded from AIR (unstable rates).",
        "",
    ]
    by_model = fairness["by_model"]
    attrs = list(next(iter(by_model.values())))
    out += _table(
        ["Model", *[f"min AIR {a}" for a in attrs], *[f"TPR gap {a}" for a in attrs]],
        [
            [
                MODEL_NAMES[k],
                *[f"{v[a]['min_air']:.3f}" for a in attrs],
                *[f"{v[a]['tpr_gap']:.3f}" for a in attrs],
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
                    f"{stats['air'][group]:.3f}",
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
        ["Alternative", "AUC", "Approval", "Bad rate of approved", "Min AIR", "TPR gap", "FPR gap"],
        [
            [
                _lda_name(r, champion["model"]),
                f"{r['auc']:.4f}",
                _pct(r["approval_rate"]),
                _pct(r["bad_rate_approved"]),
                f"{r['min_air']:.3f}",
                f"{r['tpr_gap']:.3f}",
                f"{r['fpr_gap']:.3f}",
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
            f"{_pct(ref['approval_rate'])}, min AIR {ref['min_air']:.3f}."
        )
    out += ["", "### Largest SHAP contributions (LightGBM, hold-out sample)", ""]
    out += _table(
        ["Feature", "mean |SHAP|"],
        [[f"`{r['feature']}`", r["mean_abs_shap"]] for r in m["shap_top_features"]],
    )
    out.append("")
    return out
