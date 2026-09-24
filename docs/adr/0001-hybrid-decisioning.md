# ADR 0001 — Hybrid decisioning: rules + calibrated PD + scorecard

**Status:** accepted · 2026-09

## Context
Credit decisions must be accurate, explainable to applicants and regulators, and changeable by the credit policy team without redeploying code. Pure ML is opaque for knock-out criteria (age, legal follow-up); pure rules cannot rank risk inside the eligible population.

## Decision
A three-layer engine, evaluated as a pure function of a persisted feature snapshot:
1. **Policy rules** in a versioned YAML DSL (`rules/policy_v1.yaml`) evaluated by a whitelisted AST interpreter (no `eval`). Rules either `decline` (knock-out) or `refer` (human review) and carry a reason code.
2. **Calibrated PD model** (monotone LightGBM + isotonic) drives auto-approve / grey zone / auto-decline cut-offs, the risk band and the limit factor.
3. **WoE scorecard** (optbinning) gives a transparent 300–900 point view shown next to SHAP reasons.
Limits (DSR capacity, income multiple, band factor) can turn a request into a conditional counter-offer.

## Consequences
+ Policy changes are data (versioned, back-tested, four-eyes approved).
+ Every decision is replayable (`rule_set_version`, `model_version`, `feature_hash`).
− Two model families to govern; mitigated by the model inventory and cards.
