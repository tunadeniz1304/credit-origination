# Final Report — v2.1.0

v2 answers the independent audit of v2.0.0 (23 findings, plan in `docs/PLAN_v2.md`). Every finding was
first reproduced as a strict `xfail` test in `tests/test_audit_findings_v2.py` (commit `219d621`); the
fixing commit removed the marker, so a regression turns the suite red.

## Findings

| # | Finding | Fix commit(s) | Regression test |
|---|---|---|---|
| F01 | DSR not re-checked after pricing | `bd47956` | `test_f01_offer_dsr_rechecked_with_priced_taxed_instalment` |
| F02 | Reason codes without materiality | `f2ffcb3` | `test_f02_non_material_tenure_does_not_produce_reason` |
| F03 | Constant asset correlation | `cb50011` | `test_f03_basel_other_retail_correlation` |
| F04 | Raw enum codes shown to users | `20de2bc`, `86b412b` | `test_f04_user_facing_labels_for_enum_codes` |
| F05 | Fraud signal tuned to own PDF generator | `2c4dcc4` | `test_f05_genuine_document_from_other_tool_is_not_flagged` |
| F06 | OCR path not really working | `7dc52e8`, `977512d`, `ed9e23d` | `test_f06_ocr_status_reported_in_health`, OCR test in CI |
| F07 | Fairness good by construction; LDA table at different approval rates | `cb309f9`, `569ca14`, `41c6f8c` | `test_f07_*`, `tests/test_fairness.py` |
| F08 | Champion/challenger without statistical evidence | `eb9c0cd`, `e44801b`, `383a843` | `test_f08_champion_challenger_shows_statistical_evidence` |
| F09 | SQLite "database is locked" 500s | `fec5fb4`, `12cc836` | `test_f09_*`, `tests/test_concurrency.py` |
| F10 | Circuit breaker not atomic, unbounded HALF_OPEN probes | `009eb45` | `test_f10_half_open_admits_a_single_probe` |
| F11 | Duplicate pending reviews under a race | `e84dcbb` | `test_f11_single_pending_review_per_application` |
| F12 | p50 15.9 s | `12cc836` | `docs/PERFORMANCE.md` (before / after) |
| F13 | CSP `unsafe-eval` | `49ea97a` | `test_f13_csp_without_unsafe_eval` |
| F14 | JWT in web storage | `0b52b0a`, `be43299` | `test_f14_login_sets_httponly_cookie_and_csrf_is_enforced` |
| F15 | Demo users / open registration in prod | `0b52b0a` | `test_f15_prod_defaults_disable_demo_users` |
| F16 | KVKK redaction misses Turkish İ/ı | `30a1452` | `test_f16_redaction_handles_turkish_dotted_i` |
| F17 | Queue without pagination / search | `09e6f50`, `dbbad92` | `test_f17_queue_is_paginated` |
| F18 | Accessibility | `f510daf` | `test_f18_every_form_control_is_labelled` |
| F19 | No browser E2E tests | `5e91536`, `74d103c` | `tests/e2e/` (Playwright) |
| F20 | "SR 26-2" citation suspected invented | `5eba20b` | `test_f20_regulatory_citations_are_verified_with_sources` |
| F21 | Vendor parity table in README | `53f2272` | `test_f21_readme_has_limitations_instead_of_vendor_parity_table` |
| F22 | Internal LLM host in tracked files | `91d73c2` | `test_f22_internal_llm_host_not_in_tracked_files` |
| F23 | This report | this commit | `test_f23_final_report_v2_exists` |

**F20 note.** The audit assumed SR 26-2 did not exist. It does (Fed SR 26-2 / OCC Bulletin 2026-13,
17 April 2026, superseding SR 11-7). The citation was kept and every regulatory reference now has an
official source in `docs/COMPLIANCE.md`; CFPB Circular 2022-03 is marked as withdrawn (12 May 2025).

## Lane A — real-data validation

Public data with pinned checksums (`docs/DATA.md`): UCI Taiwan default of credit card clients (30,000)
and Statlog German Credit (1,000). Stratified 5-fold CV plus a 20 % hold-out; bootstrap 95 % AUC CIs;
own DeLong implementation (checked against hand computation and sklearn AUC); ECE, Hosmer–Lemeshow and decile calibration
(deciles on average ranks, so tied PDs stay in one decile).

| Hold-out AUC [95 % CI] | UCI Taiwan | German Credit |
|---|---|---|
| Monotone LightGBM + isotonic | 0.779 [0.765, 0.792] | 0.770 [0.698, 0.838] |
| WoE scorecard | 0.767 [0.752, 0.781] | 0.779 [0.707, 0.847] |
| Logistic regression | 0.758 [0.743, 0.773] | 0.758 [0.684, 0.827] |

Full report: `docs/VALIDATION_REPORT.md`; machine-readable: `artifacts/validation/*/metrics.json`,
served by `GET /api/v1/models/validation` and shown in the UI's model validation tab.

**Lane B (anchoring, not validation).** Taiwan repayment status is mapped onto the platform's bureau
features; a behaviour sub-score learnt on real defaults feeds the production model; the synthetic
generator is anchored so that the default rate per delinquency band matches the real curve (11.7 /
25.0 / 43.5 / 62.9 %, tolerance test ±5 points). The PD level is imposed from a real proxy curve
(anchoring), not validated: the anchor is next-month card default, used as the level of a 12-month
90+DPD loan PD. Real validation is lane A (methodology) plus the behaviour sub-score. The PD scale
therefore moved up and the policy/authority cut-offs were recalibrated; on the anchored synthetic
test set (PD thresholds only, `policy_cutoffs` in `artifacts/validation/lane_b.json`) the realised
bad rate of the auto-approved book is 4.0 % under v2 (55 % auto-approved, 19 % auto-declined). This
is a **risk-appetite change that needs credit committee sign-off**: on the same v2 model the v1
cut-offs give 2.7 % (40 % auto-approved), so v2 loosens the appetite by about +50 % relative. The
retired v1 model with the v1 cut-offs shows 4.8 % (59 % auto-approved), but it is not a like-for-like
baseline: it under-predicts that book about 3.7× (mean PD 1.3 %). On the same population the lowest PD decile is
under-predicted (observed/predicted 1.99; 752 applicants, 12 defaults) — an open finding.

## Champion

Rule in `rules/validation.yaml`: start from the simplest family; a more complex one wins only with
DeLong p < 0.05 **and** ΔAUC ≥ 0.005 on the pooled **out-of-fold** CV predictions of the training part;
the hold-out only confirms the choice.

* **UCI Taiwan:** LightGBM beats the scorecard out-of-fold by +0.014 (p = 6.8e-18) → **LightGBM is
  champion**; hold-out confirmation +0.012 (p = 5.1e-5). Its low-risk deciles are well calibrated
  (observed/predicted 1.05, ECE 0.012, HL p = 0.42).
* **German Credit:** LightGBM is significantly weaker than the scorecard out-of-fold (ΔAUC −0.020,
  p = 0.026; hold-out −0.009, p = 0.63) → scorecard. Recorded as a caveat: the boosting advantage
  depends on sample size.
* Model promotion through the governance API needs real-data evidence for the challenger's model
  family on every dataset: refused if it is significantly worse on any set on either basis
  (out-of-fold or hold-out DeLong), if evidence is missing, or if the family is unknown (no borrowed
  evidence). A LightGBM challenger to the scorecard is therefore refused (German out-of-fold ΔAUC
  −0.020, p = 0.026). A challenger of the champion's own family (a retrain, the most common
  promotion) is allowed by the gate with `same_family: true` — family-level evidence cannot tell two
  artifacts of one family apart, so the artifact-level comparison and four-eyes approval decide. The
  evidence is family-level; the artifact is
  trained on synthetic data, so evidence tied to its hash is impossible. Four-eyes still applies.

## Fairness

Compared at the **same approval rate** for every model (`approve_at_rate`; tied PDs at the cut-off
are approved in a seeded random order and each minimum AIR is reported with its range over 20 seeds —
a range that covers **only** the tie-breaking), with a 95 % stratified bootstrap interval for the
minimum AIR (1,000 resamples within each protected group, decisions held fixed) for the sampling
error of the hold-out, on real data:

* **UCI Taiwan:** every attribute passes the four-fifths rule with the whole interval above 0.80
  (worst: education, AIR 0.877, 95 % CI [0.834, 0.922], seeds 0.874–0.882).
* **German Credit:** the champion's age-band AIR is **0.698 — below 0.80 but indicative, not
  statistically established**: on the 200-row hold-out the groups have 78/66/34/22 rows (the 50+
  reference group 22, 19 approved) and the 95 % CI [0.545, 0.845] contains 0.80. The
  less-discriminatory-alternative search includes the other trained families and requires the AUC
  loss to stay within 0.010 both out-of-fold and on the hold-out; none qualifies (LightGBM would reach
  AIR 0.780, still below 0.80, but loses 0.020 AUC out-of-fold against 0.009 on the hold-out, and its
  paired AIR gain CI [−0.105, +0.187] contains 0). This is an
  **open model-risk finding** for the model risk committee to confirm on more data, not fixed and not hidden (`docs/FAIRNESS_REPORT.md`, `docs/MODEL_CARD.md`).
  `FOREIGN_WORKER` is not testable (one group only above the minimum size) and is reported as n/a.
* The synthetic generator now contains proxy correlations (tenure ↔ age, income ↔ province/gender), so
  the synthetic fairness results are no longer good by construction.

## Performance

Details in `docs/PERFORMANCE.md`.

| Inline mode (SQLite) | v2.0.0 | v2.1.0 |
|---|---|---|
| One clean application, p50 / p95 | 1.6 s / 1.7 s | 1.6 s / 1.7 s |
| 40 applications, 20 concurrent clients | did not finish in 12 min (writer deadlock) | 40 / 40 in 35 s, 0 errors |

The p50 < 5 s target is met. The audit's 15.9 s was a p50 under load. The v2.1 gain is that inline mode
no longer deadlocks under concurrency; `tests/test_concurrency.py` (20 applications + 20 logins) has
0 HTTP 500s.

## Quality gate

`ruff check`, `ruff format --check`, `mypy app`, `pytest --cov=app` (total ≥ 90 %; governance, worker and
task dispatcher ≥ 80 %), `docker compose build`, GitHub Actions CI (with Tesseract + Turkish model and
Playwright).

## Remaining limitations

* Production features remain Turkish **synthetic** data; only the methodology and the behaviour
  sub-score are validated on real (non-Turkish) data. The PD level is imposed from a real proxy curve
  (anchoring), not validated.
* No reject inference and no real out-of-time test (the public sets have no usable time axis).
* The German Credit age-band fairness finding is open and only indicative (95 % CI of the minimum AIR
  [0.545, 0.845] contains 0.80 on the 200-row hold-out; no less discriminatory alternative within the
  AUC-loss limit on both bases).
* The v2 policy cut-offs loosen risk appetite and are pending credit committee sign-off.
* Inline/SQLite is a single-writer mode; throughput needs the Docker topology.

## Audit rounds

An independent sceptical reviewer (senior engineer + credit-risk expert, separate agent, read-only)
audited the repository; findings were fixed and the next round re-audited the result.

### Round 1 — score 7.5 / 10, 13 findings

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | high | A cancelled half-open probe never freed its slot; the breaker could stay HALF_OPEN forever | Slot released on `BaseException`, probe slots are leases (`6adecf4`) |
| 2 | high | Prod started with the public development JWT secret | Prod refuses to start with a weak JWT secret, empty PII/blind-index keys or registration without CAPTCHA (`e544286`) |
| 3 | high | Lane B "calibration validated on real data" is circular | Reworded everywhere as anchoring, not validation (`d93ab8e`) |
| 4 | high | Policy recalibration numbers not reproducible | `run_lane_b.py --policy-only` writes `policy_cutoffs` to `lane_b.json`; test ties docs to it (`c739fb0`) |
| 5 | medium | Promotion gate judged by family on one dataset; `ebm` borrowed logistic evidence | Unknown families refused; challenger must not be significantly worse on any dataset; scope stated (`af15e67`) |
| 6 | medium | Isotonic ties made AIR depend on row order | Seeded tie-breaking, AIR range over 20 seeds, tie-aware deciles (`af15e67`) |
| 7 | medium | Champion chosen and reported on the same hold-out | Selection on out-of-fold CV, hold-out confirms (`af15e67`) |
| 8 | medium | Order-dependent flaky tests | Tests create their own data (`9f6627e`, `ae477b4`) |
| 9 | medium | Coverage gate overstated | CI `--cov-fail-under=90` + `scripts/check_coverage.py` floors (`1d19ca9`) |
| 10 | low | F05 test could never fail; F01 one persona | F05 checks current signal names and a positive case; F01 three personas (`1d19ca9`) |
| 11 | low | LDA search ignored other trained families | Included (`af15e67`) |
| 12 | low | Untestable attribute shown as passing | Reported as n/a (`af15e67`) |
| 13 | low | Registration on in prod; logout left JWT valid | Registration off by default in prod; logout revokes the token id (`e544286`) |

New open finding surfaced by the fixes: with tie-aware deciles the lowest PD decile of the anchored
synthetic population is under-predicted (observed/predicted 1.99, 12 defaults in 752).

### Round 2 — score 8.3 / 10, 9 findings

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | medium | Order-dependent tests (shared grey-zone application; Celery test reaching a real broker) | Function-scoped fixture; dispatcher registry routed to the fake (`7a23f2c`) |
| 2 | high | Promotion and LDA judged on the hold-out while selection used out-of-fold evidence; scorecard → LightGBM promotion was allowed, German LDA "within limit" only on the hold-out | Promotion refused if significantly worse on either basis; LDA loss must be within 0.010 on both — German now has **no** qualifying alternative (`1b4bc45`) |
| 3 | high | Policy v2 described as "same risk appetite" although it loosens it on the same model (2.7 % → 4.0 %) | Stated as a risk-appetite change pending credit committee sign-off, with both comparisons (`1b4bc45`) |
| 4 | medium | Straggler outcomes: a late success closed an OPEN breaker, late failures moved `opened_at` | Every call carries a ticket (closed epoch / probe lease id); outcomes apply only to the state they were admitted under (`9130202`) |
| 5 | medium | A stale probe could release another probe's slot | Probe slots are per-lease ids; only the matching lease is released (`9130202`) |
| 6 | medium | HTTP 4xx counted as service failures | Only 5xx, transport and other errors count (`9130202`) |
| 7 | medium | Revocation list: "no Redis" cached forever, Redis error → 500, no own setting | `SESSION_REVOCATION_BACKEND`, fails closed (401 / logout 503), re-probes Redis, required in prod (`d941ace`) |
| 8 | medium | Prod gaps: demo users could be forced on, worker skipped the checks, dev Fernet fallback, stale roles in tokens | Prod refuses `SEED_DEMO_USERS`; worker runs the same checks; crypto refuses the dev key in prod; tokens of deactivated users or changed roles stop working (`d941ace`) |
| 9 | low | Lane B policy table not bound to the policy file or model | `policy_cutoffs` records thresholds and model SHA-256; test compares them with `policy_v2.yaml` and the committed model (`1b4bc45`) |

