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

## Real-data validation and champion decision

The synthetic metrics below are *not* evidence of real-world performance: the target was produced by
the author's own formula. The modelling recipe was therefore validated on real, labelled public
default data (`docs/VALIDATION_REPORT.md`, lane A; data provenance in `docs/DATA.md`).

The champion is chosen by evidence, not assumed. Rule (`rules/validation.yaml`): start from the
simplest family (scorecard < logistic regression < LightGBM); a more complex family replaces it only
if the DeLong test on the stratified hold-out is significant (p < 0.05) **and** the AUC gain is
material (≥ 0.005).

| Evidence (hold-out) | UCI Taiwan (30,000) | German Credit (1,000) |
|---|---|---|
| LightGBM AUC [95 % bootstrap CI] | 0.779 [0.765, 0.792] | 0.770 [0.698, 0.838] |
| Logistic regression AUC | 0.758 [0.743, 0.773] | 0.758 [0.684, 0.827] |
| WoE scorecard AUC | 0.767 [0.752, 0.781] | 0.779 [0.707, 0.847] |
| LightGBM vs scorecard | ΔAUC +0.012, DeLong p = 5.1e-5 → **selected** | ΔAUC −0.009, p = 0.63 → not significant |
| Champion | **Monotone LightGBM + isotonic** | WoE scorecard |
| Low-risk deciles observed / predicted | 1.04 | 1.75 (n = 60, unstable) |

**Decision.** The production PD champion stays a monotone LightGBM because on the large real set it
beats both simpler families significantly and materially, and its isotonic calibration holds in the
low-risk deciles (observed/predicted 1.04; ECE 0.010; Hosmer–Lemeshow p = 0.64). On the small German
set the simpler scorecard is at least as good, which is recorded as a caveat: the advantage of
gradient boosting depends on sample size. The logistic-regression challenger is significantly weaker
on real data (ΔAUC −0.021, p = 4.4e-7), so promotion requests for it are refused by the governance
API (`approve_promotion` checks the committed evidence; four-eyes still applies).

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
