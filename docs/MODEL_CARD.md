# Model Card — pd_lgbm_v1 (champion PD model)

A live, per-model card is also served by `GET /api/v1/models/{id}/card?format=json|md|pdf`.

| Item | Value |
|---|---|
| Purpose | Probability of 90+ days past due within 12 months for retail personal/auto loans |
| Type | LightGBM, monotone constraints, isotonic calibration (`artifacts/models/pd_lgbm_v1.*`) |
| Owner / approval | Model risk committee (`model_yoneticisi` role), four-eyes promotion |
| Intended use | Input to the hybrid engine (cut-offs, risk band, limit factor, pricing) under human oversight; never the sole basis of an adverse decision without the right to object |
| Out of scope | Business loans, collateral valuation, collections |

## Data
Seeded synthetic population of 50,000 applications (`scripts/generate_training_data.py`, seed 20260924) drawn from the same persona mix as the mock providers: clean salaried, grey zone, over-indebted, delinquent, thin file, self-employed. Target rate ≈ 10.8 %. Time-based split: months 1–16 train, 17–20 calibration, 21–24 test. **No real personal data.**

## Features (18) and monotonicity
Bureau score (↓), bureau hit, arrears count (↑), max days past due (↑), inquiries (↑), active loans (↑), utilisation (↑), DSR (↑), loan-to-income (↑), term (↑), log income (↓), employment months (↓), income volatility (↑), negative-balance days (↑), NSF count (↑), gambling share (↑), savings rate (↓), average balance / income (↓).
**Excluded:** gender, age band, province (monitoring only); age is used solely by the legal eligibility rule.

## Performance (time-based test set, n = 8,208)

| Metric | pd_lgbm_v1 | scorecard_woe_v1 | challenger_lr_v1 |
|---|---|---|---|
| AUC | 0.895 | 0.892 | 0.896 |
| Gini | 0.790 | 0.783 | 0.791 |
| KS | 0.625 | 0.619 | 0.618 |
| Brier | 0.0628 | 0.0639 | 0.0625 |

Calibration plot: `docs/img/calibration.png` (decile table in the model metadata).

## Fairness
Minimum adverse impact ratio at the auto-approval cut-off: gender 0.99, age band 0.97, province 0.93 (all ≥ 0.8). Details and the less-discriminatory-alternative search: `docs/FAIRNESS_REPORT.md`.

## Explainability
SHAP TreeExplainer on the log-odds margin → top four adverse contributions mapped to Turkish reason codes (`rules/reason_codes.yaml`) with the applicant's values; scorecard points lost are shown alongside; counterfactuals search only mutable levers (amount, term, closing existing debt).

## Limitations
* Synthetic data is more separable than real portfolios; retrain and validate on bank data before production.
* Macroeconomic shocks shift distributions — PSI drift is monitored hourly (`/api/v1/governance/drift`).
* Thin-file applicants without open-banking consent carry more uncertainty (routed via the grey zone).

## Monitoring & change management
PSI/CSI drift, live AIR, champion/challenger agreement, replay tests for determinism, two-person approval for promotion and for rule-set activation (with backtest).
