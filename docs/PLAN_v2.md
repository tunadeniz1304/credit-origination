# Plan v2 — close the audit findings and validate the methodology on real public data

v1 (tag `v2.0.0`) scored **7.5/10** in an independent, sceptical audit. The main criticisms were
that the model was trained only on data produced by the author's own formula (self-confirming
metrics), that the champion could not beat its own challenger, that fairness numbers were good by
construction, that DSR and pricing disagreed, and that reason codes could be meaningless.

This plan closes every finding with a regression test and adds a real-data validation lane.

## Baseline (before v2)

| Gate | Result |
|---|---|
| Tests | 231 passed |
| Coverage | 89 % total; `governance/fairness.py` 26 %, `worker/tasks.py` 38 %, `core/task_dispatcher.py` 50 % |
| ruff / ruff format / mypy | clean |

## V0 — reproductions

`tests/test_audit_findings_v2.py` reproduces each finding **before** its fix. Every reproduction is
marked `xfail(strict=True)`; a strict xfail fails the suite if the test unexpectedly passes, so the
red state is recorded without breaking the gate. The fix commit removes the marker.

Reproduction evidence gathered while writing the tests:

* **F01** `temiz`, 45,000 TL income, 800,000 TL / 36 m: the engine reports `dsr_offer = 0.4999`,
  but the priced instalment (contract rate + BSMV + KKDF) is 21,695 TL, i.e. a real DSR of
  **0.5496 > 0.50** cap. The same gap appears for the `gri` and `asiri_borclu` personas.
* **F02** the same file (PD 1.2 %, 52 months of tenure) lists `R05_ISTIHDAM_KISA`.
* **F05** a genuine e-Devlet PDF whose producer is iText (not reportlab) scores `producer_mismatch`.
* **F16** a name registered as `İsmail Işık` leaks the surname when the text contains the
  ASCII-folded `ISMAIL ISIK` (and vice versa).
* **F12** (performance) and the 20 + 20 concurrency scenario of **F09** are measured with
  `scripts/load_test.py` / `scripts/profile_pipeline.py` rather than a unit test; the numbers go to
  `docs/PERFORMANCE.md`.

## Findings → phase → regression test

| # | Finding | Phase | Test |
|---|---|---|---|
| F01 | DSR not re-checked after pricing | V4 | `test_f01_offer_dsr_rechecked_with_priced_taxed_instalment` |
| F02 | No materiality threshold for reason codes | V4 | `test_f02_non_material_tenure_does_not_produce_reason` |
| F03 | Constant asset correlation 0.15 | V4 | `test_f03_basel_other_retail_correlation` |
| F04 | Raw enum codes in letters / PDF | V4 | `test_f04_user_facing_labels_for_enum_codes` |
| F05 | `producer_mismatch` tuned to own generator | V5 | `test_f05_genuine_document_from_other_tool_is_not_flagged` |
| F06 | OCR path not really working | V5 | `test_f06_ocr_status_reported_in_health` |
| F07 | Fairness good by construction, 26 % coverage, broken LDA table | V2/V4 | `test_f07_*` |
| F08 | Champion/challenger without statistical evidence | V2/V4 | `test_f08_champion_challenger_shows_statistical_evidence` |
| F09 | SQLite "database is locked" 500s | V6 | `test_f09_sqlite_uses_wal_and_normal_sync` + concurrency test |
| F10 | Breaker not atomic, unlimited HALF_OPEN probes | V6 | `test_f10_half_open_admits_a_single_probe` |
| F11 | Duplicate pending review race | V6 | `test_f11_single_pending_review_per_application` |
| F12 | p50 15.9 s end-to-end | V6 | `docs/PERFORMANCE.md` (before / after) |
| F13 | CSP `unsafe-eval` | V6 | `test_f13_csp_without_unsafe_eval` |
| F14 | JWT in web storage | V6 | `test_f14_login_sets_httponly_cookie_and_csrf_is_enforced` |
| F15 | Demo users / open registration in prod | V6 | `test_f15_prod_defaults_disable_demo_users` |
| F16 | KVKK redaction misses Turkish İ/ı variants | V6 | `test_f16_redaction_handles_turkish_dotted_i` |
| F17 | Queue without pagination / search | V7 | `test_f17_queue_is_paginated` |
| F18 | Accessibility (labels, keyboard, contrast, chart tables) | V7 | `test_f18_every_form_control_is_labelled` |
| F19 | No browser E2E tests | V7 | `tests/e2e/` (pytest-playwright) |
| F20 | Unverifiable "SR 26-2" citation | V8 | `test_f20_no_unverified_sr_26_2_citation` |
| F21 | Vendor parity table in README | V8 | `test_f21_readme_has_limitations_instead_of_vendor_parity_table` |
| F22 | Internal LLM host in tracked files | V8 | `test_f22_internal_llm_host_not_in_tracked_files` |
| F23 | Final report v2 | V8 | `test_f23_final_report_v2_exists` |

## Real-data validation design

**Lane A — methodology proof.** The unchanged training recipe (monotone LightGBM, logistic
regression, optbinning scorecard, isotonic calibration, SHAP, fairness) runs on the UCI Taiwan set
with its *own* features: stratified 5-fold CV on 80 % plus a 20 % stratified holdout (there is no
time axis). Metrics: AUC with bootstrap 95 % CI, Gini, KS, Brier, log loss, DeLong test between
models (own `scipy` implementation, checked against a textbook example), decile calibration,
Hosmer–Lemeshow and ECE with the low-risk deciles reported separately. The champion is chosen by
evidence: LightGBM only if its AUC gain over LR is significant (DeLong) **and** material (config
threshold); otherwise the simpler model wins. Fairness uses the real SEX / AGE band / EDUCATION /
MARRIAGE attributes; every model and mitigation is compared at the **same approval rate**. German
Credit is a second, small check.

**Lane B — anchoring the production model.** `app/decisioning/public_mapping.py` maps the Taiwan
payment history, utilisation and payment ratio onto bureau-style behaviour features; a
`bureau_behavior_score` is trained on real data and becomes an input of the production PD model;
the synthetic generator is re-calibrated so that the default rate per delinquency band matches the
real curve within a configured tolerance; the PD level and low-risk deciles are anchored to real
observed rates.

## Phase exit criteria

Every phase ends with `ruff check`, `ruff format
--check`, `mypy app`, `pytest --cov=app` (≥ 90 % total; `governance/`, `worker/`,
`core/task_dispatcher.py` ≥ 80 % each), `docker compose build`, commit and push.
