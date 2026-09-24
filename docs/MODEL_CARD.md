# Model Card — pd_lgbm_v2 (champion PD model)

A live, per-model card is also served by `GET /api/v1/models/{id}/card?format=json|md|pdf`.

| Item | Value |
|---|---|
| Purpose | Probability of 90+ days past due within 12 months for retail personal/auto loans |
| Type | LightGBM, monotone constraints, isotonic calibration (`artifacts/models/pd_lgbm_v2.*`) |
| Owner / approval | Model risk committee (`model_yoneticisi` role), four-eyes promotion |
| Intended use | Input to the hybrid engine (cut-offs, risk band, limit factor, pricing) under human oversight; never the sole basis of an adverse decision without the right to object |
| Out of scope | Business loans, collateral valuation, collections |

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

## Honest scope

The production model works on **Turkey-specific synthetic features** (KKB score, DSR, open-banking
cash flow). The **methodology, the behaviour sub-score and the calibration are validated on real public
data**. A real bank portfolio is required for re-training and independent validation before any
production use.

## Data (production model)

Seeded synthetic population of 100,000 applications (`scripts/run_lane_b.py`, seed 20260924) drawn
from the persona mix of the mock providers. Since v2 the generator is **anchored to real data**
(lane B): bureau behaviour enters through `bureau_behavior_score`, learnt on real UCI Taiwan defaults,
and the default rate within each delinquency band equals the real rate (11.7 % / 25.0 % / 43.5 % /
62.9 % for 0 / 1 / 2 / 3+ months of arrears; synthetic within ±1 point, tolerance ±5 points). Proxy
correlations were added (tenure ↔ age, income ↔ province and gender). Target: 90+ DPD within 12 months;
synthetic default rate 16.3 % (v1: 10.8 %, which under-stated risk about twofold in the low-risk
region). Time-based split: months 1–16 train, 17–20 calibration, 21–24 test. **No real personal data.**

## Features (19) and monotonicity
Bureau score (↓), bureau hit, arrears count (↑), max days past due (↑), inquiries (↑), active loans (↑),
utilisation (↑), **bureau behaviour score (↑, real-data sub-score)**, DSR (↑), loan-to-income (↑),
term (↑), log income (↓), employment months (↓), income volatility (↑), negative-balance days (↑), NSF
count (↑), gambling share (↑), savings rate (↓), average balance / income (↓).
**Excluded:** gender, age band, province (monitoring only); age is used solely by the legal
eligibility rule.

## Performance (synthetic time-based test set, n = 16,803)

| Metric | pd_lgbm_v2 | scorecard_woe_v2 | challenger_lr_v2 |
|---|---|---|---|
| AUC | 0.827 | 0.823 | 0.825 |
| Gini | 0.653 | 0.645 | 0.650 |
| KS | 0.511 | 0.508 | 0.508 |
| Brier | 0.1065 | 0.1085 | 0.1072 |

These are synthetic-data numbers: the three families are within 0.005 AUC of each other here, which
is why the champion choice rests on the **real-data** evidence above, not on this table. Calibration
(lane B, synthetic test set): ECE 0.005, Hosmer–Lemeshow p = 0.18, lowest three deciles
observed/predicted 1.19 / 1.09 / 0.90 (tolerance ±25 %); mean predicted PD per delinquency band
11.7 % / 23.8 % / 49.7 % / 63.7 % vs real 11.7 % / 25.0 % / 43.5 % / 62.9 %. Plot:
`docs/img/calibration.png`.

## Policy recalibration (policy_v2, authority_v2)

Because the PD scale moved to real levels, the v1 cut-offs no longer expressed the same risk appetite.
Risk appetite is defined as the realised bad rate of the automatically approved book: 4.8 % under v1
(old model, anchored population). v2 keeps it at or below that level: auto-approve PD ≤ 8 % (55 % of
applicants, realised bad rate 4.0 %), auto-decline PD ≥ 30 % (19 %), specialist grey zone in between;
risk bands A ≤ 3 %, B ≤ 8 %, C ≤ 15 %, D ≤ 30 %. The authority matrix PD limits moved accordingly
(specialist ≤ 15 %, senior ≤ 30 %, four-eyes above 20 %). Both files are versioned; the RAG policy text
was updated with them.

## Fairness
Real-data fairness (UCI Taiwan, German Credit) at the same approval rate for every model is in
`docs/VALIDATION_REPORT.md`: on Taiwan every attribute passes the four-fifths rule (worst: education
0.878); on German Credit the scorecard champion **fails for age band (0.698)** and no alternative within
the allowed AUC loss fixes it — recorded as an open model-risk finding. Synthetic results:
`docs/FAIRNESS_REPORT.md`.

## Explainability
SHAP TreeExplainer on the log-odds margin → adverse contributions mapped to Turkish reason codes (`rules/reason_codes.yaml`) with the applicant's values. A model reason is shown only if it is material (SHAP ≥ 0.10 log-odds and ≥ 5 scorecard points lost) **and** the value is in the feature's adverse range (e.g. tenure < 24 months); at most four codes; approved files get "improvement area" wording, never decline language. Counterfactuals search only mutable levers (amount, term, closing existing debt).

## Limitations
* Production features are synthetic; only the methodology, the behaviour sub-score and the calibration level are validated on real (non-Turkish, credit-card) data. Retrain and validate on bank data before production.
* The real anchor's target (next-month card default) differs from the platform target (90+ DPD in 12 months).
* Macroeconomic shocks shift distributions — PSI drift is monitored hourly (`/api/v1/governance/drift`).
* Thin-file applicants without open-banking consent carry more uncertainty (routed via the grey zone).

## Monitoring & change management
PSI/CSI drift, live AIR, champion/challenger agreement, replay tests for determinism, two-person approval for promotion and for rule-set activation (with backtest).
