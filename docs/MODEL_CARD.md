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
if the DeLong test is significant (p < 0.05) **and** the AUC gain is material (≥ 0.005). The test is
run on the pooled **out-of-fold** predictions of the 5-fold CV on the training part; the stratified
hold-out is used only to confirm the choice, so the hold-out AUCs below are not selection-biased.

| Evidence | UCI Taiwan (30,000) | German Credit (1,000) |
|---|---|---|
| Out-of-fold AUC: LightGBM / LR / scorecard | 0.785 / 0.757 / 0.771 | 0.776 / 0.774 / 0.796 |
| LightGBM vs scorecard (out-of-fold, selection) | ΔAUC +0.014, DeLong p = 6.8e-18 → **selected** | ΔAUC −0.020, p = 0.026 → significantly worse |
| Hold-out LightGBM AUC [95 % bootstrap CI] | 0.779 [0.765, 0.792] | 0.770 [0.698, 0.838] |
| Hold-out logistic regression AUC | 0.758 [0.743, 0.773] | 0.758 [0.684, 0.827] |
| Hold-out WoE scorecard AUC | 0.767 [0.752, 0.781] | 0.779 [0.707, 0.847] |
| LightGBM vs scorecard (hold-out, confirmation) | ΔAUC +0.012, p = 5.1e-5 | ΔAUC −0.009, p = 0.63 |
| Champion | **Monotone LightGBM + isotonic** | WoE scorecard |
| Low-risk deciles observed / predicted (hold-out) | 1.05 | 1.75 (n = 60, unstable) |

**Decision.** The production PD recipe stays a monotone LightGBM because on the large real set it
beats both simpler families significantly and materially out-of-fold, the hold-out confirms it, and
its isotonic calibration holds in the low-risk deciles (observed/predicted 1.05; ECE 0.012;
Hosmer–Lemeshow p = 0.42). On the small German set the scorecard is the champion (LightGBM is
significantly weaker out-of-fold), which is recorded as a caveat: the advantage of gradient boosting
depends on sample size.

**Promotion gate.** `approve_promotion` checks the committed real-data evidence for the model
*family* of the challenger against the champion's family on **every** real dataset and on **both**
bases — the out-of-fold CV predictions (the basis the champion is chosen on) and the hold-out:
promotion is refused if the challenger family is significantly worse on any set on either basis
(DeLong ΔAUC < 0 and p < 0.05), if evidence for a set is missing, or if the family is unknown (no
evidence is borrowed from another family). Four-eyes still applies. The logistic-regression
challenger is significantly weaker on Taiwan (hold-out ΔAUC −0.021, p = 4.4e-7), so its promotion is
refused. A LightGBM challenger to a scorecard champion is refused too: on German Credit it is
significantly weaker out-of-fold (ΔAUC −0.020, p = 0.026) even though the hold-out difference is not
significant (−0.009, p = 0.63). **Limitation:** this is evidence
for the recipe/family, not for the artifact: the production artifact is trained on synthetic data,
so real-data evidence tied to its hash is impossible; the API response states this
(`evidence_scope`).

## Honest scope

The production model works on **Turkey-specific synthetic features** (KKB score, DSR, open-banking
cash flow). Real-data validation covers only **the methodology (lane A) and the behaviour sub-score**
learnt on real defaults. The **PD level is imposed from a real proxy curve (anchoring), not
validated**: the anchor is next-month credit-card default (UCI Taiwan), used as the level of a
12-month 90+DPD personal-loan PD. A real bank portfolio is required for re-training and independent
validation before any production use.

## Data (production model)

Seeded synthetic population of 100,000 applications (`scripts/run_lane_b.py`, seed 20260924) drawn
from the persona mix of the mock providers. Since v2 the generator is **anchored to a real proxy
curve** (lane B): bureau behaviour enters through `bureau_behavior_score`, learnt on real UCI Taiwan
defaults, and the default rate within each delinquency band is set to the real next-month card
default rate (11.7 % / 25.0 % / 43.5 % / 62.9 % for 0 / 1 / 2 / 3+ months of arrears; synthetic within
±1 point, tolerance ±5 points). This imposes the PD level; it does not validate it, because the anchor
target differs from the platform target and the synthetic outcomes are drawn from the anchored curve
itself. Proxy correlations were added (tenure ↔ age, income ↔ province and gender). Target: 90+ DPD
within 12 months; synthetic default rate 16.3 % (v1 generator: 10.8 %; the change reflects the
anchor, not a measured error of v1). Time-based split: months 1–16 train, 17–20 calibration, 21–24
test. **No real personal data.**

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
on the anchored synthetic test set (lane B; internal consistency only, since the outcomes are drawn
from the same anchored curve): ECE 0.005, Hosmer–Lemeshow p = 0.09, lowest three deciles in
aggregate observed/predicted 1.08 (tolerance ±25 %: met). Deciles are formed on average ranks so
tied PDs stay together; per decile the ratios are 1.99 / 0.99 / 1.16, i.e. the **lowest decile
(752 applicants with PD ≤ 1.7 %, 12 defaults) is under-predicted (observed/predicted 1.99)** — an open finding
(the earlier per-decile pass came from splitting a 3,197-applicant tie block at PD 2.3 % by row
order). Mean predicted PD per delinquency band 11.7 % / 23.8 % / 49.7 % / 63.7 % vs anchor
11.7 % / 25.0 % / 43.5 % / 62.9 %. Plot: `docs/img/calibration.png`.

## Policy recalibration (policy_v2, authority_v2)

Because the PD scale moved to the anchored level, the v1 cut-offs no longer expressed the same risk
appetite. Risk appetite is defined as the realised bad rate of the automatically approved book. The
table below is produced by `scripts/run_lane_b.py` (`policy_cutoffs` in
`artifacts/validation/lane_b.json`) and is **measured on the anchored synthetic population**
(test months 21–24, n = 16,803), applying only the PD thresholds (rule-based referrals ignored), so it
inherits the proxy PD level and is not real-portfolio evidence.

| Policy | Model | Auto-approve | Grey zone | Auto-decline | Bad rate of auto-approved |
|---|---|---|---|---|---|
| v1 (PD ≤ 5 % / ≥ 20 %) | pd_lgbm_v1 (retired) | 59.1 % | 29.9 % | 11.0 % | 4.8 % |
| v1 cut-offs on the v2 model | pd_lgbm_v2 | 39.8 % | 28.0 % | 32.2 % | 2.7 % |
| **v2 (PD ≤ 8 % / ≥ 30 %)** | pd_lgbm_v2 | **54.8 %** | 25.8 % | **19.4 %** | **4.0 %** |

**v2 is a risk-appetite change and needs credit committee sign-off.** The like-for-like comparison
is on the same v2 model: the v1 cut-offs give an auto-approved book with a 2.7 % bad rate (40 %
auto-approved), the v2 cut-offs 4.0 % (55 % auto-approved) — about +50 % relative, i.e. the v2
cut-offs *loosen* the risk appetite rather than re-express the v1 appetite on the new scale. The
retired v1 model with the v1 cut-offs shows 4.8 %, but that is not a like-for-like baseline: on this
population it under-predicts the auto-approved book about 3.7× (mean PD 1.3 %, realised 4.8 %). Risk bands A ≤ 3 %, B ≤ 8 %, C ≤ 15 %, D ≤ 30 %. The authority matrix PD limits
moved accordingly (specialist ≤ 15 %, senior ≤ 30 %, four-eyes above 20 %). Both files are versioned;
the RAG policy text was updated with them.

## Fairness
Real-data fairness (UCI Taiwan, German Credit) at the same approval rate for every model is in
`docs/VALIDATION_REPORT.md`. Tied PDs at the cut-off are approved in a seeded random order (never row
order) and every minimum AIR is reported with its spread over 20 tie-break seeds. On Taiwan every
attribute passes the four-fifths rule (worst: education 0.877, seeds 0.874–0.882). On German Credit
the scorecard champion **fails for age band (0.698; no ties, so no spread)** — an open model-risk
finding. The LDA search also considers the other trained families and requires the AUC loss to be
within the 0.010 limit both out-of-fold and on the hold-out. No alternative qualifies: LightGBM
would raise the age-band AIR to 0.780 (still below 0.80) for a hold-out loss of 0.009 but an
out-of-fold loss of 0.020; the proxy-weakened logistic regression (AIR 0.732) costs 0.021 / 0.018.
The finding goes to the model risk committee with no less discriminatory alternative within the
limit.
German `FOREIGN_WORKER` is **not testable** (only one group meets the minimum group size) and is
reported as n/a, not as a pass. Synthetic results: `docs/FAIRNESS_REPORT.md`.

## Explainability
SHAP TreeExplainer on the log-odds margin → adverse contributions mapped to Turkish reason codes (`rules/reason_codes.yaml`) with the applicant's values. A model reason is shown only if it is material (SHAP ≥ 0.10 log-odds and ≥ 5 scorecard points lost) **and** the value is in the feature's adverse range (e.g. tenure < 24 months); at most four codes; approved files get "improvement area" wording, never decline language. Counterfactuals search only mutable levers (amount, term, closing existing debt).

## Limitations
* Production features are synthetic; only the methodology (lane A) and the behaviour sub-score are validated on real (non-Turkish, credit-card) data. The PD level is imposed from a real proxy curve (anchoring), not validated. Retrain and validate on bank data before production.
* The real anchor's target (next-month card default) differs from the platform target (90+ DPD in 12 months).
* Promotion evidence is family-level: the production artifact is trained on synthetic data, so no real-data evidence can be tied to its hash.
* The lowest lane B decile is under-predicted (observed/predicted 1.99) on the anchored synthetic population (see Performance).
* **Reject inference is not addressed.** A live model only observes outcomes of approved applicants, so training on booked loans biases PD downward for the declined region. Neither public set contains rejected applicants; with a real portfolio, parceling or fuzzy augmentation (or a small randomised approval band) is needed before retraining.
* **No out-of-time evidence on real data.** The public sets have no usable time axis (Taiwan is one six-month window; German Credit is undated), so lane A uses stratified CV + hold-out; PSI drift monitoring stands in for out-of-time testing in production.
* Macroeconomic shocks shift distributions — PSI drift is monitored hourly (`/api/v1/governance/drift`).
* Thin-file applicants without open-banking consent carry more uncertainty (routed via the grey zone).

## Monitoring & change management
PSI/CSI drift, live AIR, champion/challenger agreement, replay tests for determinism, two-person approval for promotion and for rule-set activation (with backtest).
