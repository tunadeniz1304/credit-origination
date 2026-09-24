# ADR 0002 — Monotone gradient boosting plus a WoE scorecard

**Status:** accepted

## Context
Regulators and applicants expect that "more debt never lowers risk" and "a higher bureau score never raises risk". Unconstrained GBMs can violate this locally, which produces contradictory reason codes.

## Decision
Train LightGBM with `monotone_constraints` for every feature with a known direction (bureau score ↓, DSR ↑, arrears ↑, tenure ↓, savings ↓ …), a time-based split (months 1–16 train, 17–20 calibration, 21–24 test) and isotonic calibration. Keep an optbinning WoE logistic scorecard with monotone bins as the regulator-friendly companion, and a logistic challenger (EBM when `interpret` is installed) that shadow-scores every decision.

## Consequences
+ SHAP contributions are directionally consistent with the reason-code dictionary.
+ Scorecard points-lost give a second, human-readable explanation.
− Slight AUC cost versus an unconstrained model (accepted for explainability).
