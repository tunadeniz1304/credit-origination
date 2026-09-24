# Final Report — Anil2 v2.0.0

Evidence for the Definition of Done of the transformation brief.

## 1. One-command platform
* `docker compose up --build` starts **postgres, redis, mailhog, app, worker, beat**; postgres/redis/app/worker report *healthy* (compose health checks). The app runs `alembic upgrade head` on start.
* `python scripts/smoke.py` against the compose stack (live DeepSeek) passes all 19 steps: readiness, clean applicant → offer → acceptance → contract → disbursal, PDF report, AI memo (`mode=live`), grey zone → maker submits → checker approves (four-eyes), delinquent → automatic rejection → KVKK objection, governance endpoints, audit-chain verification, UI and Prometheus metrics.
* Browser E2E with Playwright (`scripts/ui_screenshots.py`): applicant portal, workbench (SHAP tab) and dashboard render without console errors; screenshots in `docs/img/`.

## 2. LLM contract
* Without a key: demo mode produces professional Turkish memoranda, committee summaries and letters; `[mock-llm]` no longer exists anywhere (asserted in tests).
* With the key in `.env`: start-up log `LLM: CANLI (deepseek-v4-flash @ llm-gateway.example.org)`; `python scripts/llm_smoke.py` → `OK model=deepseek/deepseek-v4-flash latency=2183ms`.
* All three chain steps run live and pass the citation/number guard (verified against the real endpoint after adding the empty-content budget retry).
* Redaction: tests assert that no TCKN, name or phone appears in outbound LLM requests and that logs mask TCKN/IBAN (`tests/test_llm.py`). The key is never logged or returned (`/api/v1/llm/status` exposes only `key_present`).

## 3. Six demo personas
`tests/test_demo_personas.py` seeds all scenarios through the real pipeline and asserts: clean → offer; thin file → approval via cash flow; high DSR → conditional counter-offer + counterfactual; tampered payslip → `R11` + `SUPHELI` document; shared phone/IBAN → ring graph with ≥3 applicants; grey zone → AI memo + pending four-eyes review.

## 4. Models and governance
* PD model (time-based test, n = 8,208): AUC 0.895, Gini 0.790, KS 0.625, Brier 0.0628; scorecard AUC 0.892; challenger AUC 0.896 (`artifacts/models/metrics.json`, `docs/img/calibration.png`).
* Model card (JSON/MD/PDF endpoint + `docs/MODEL_CARD.md`), fairness report with min AIR 0.99 / 0.97 / 0.93 (`docs/FAIRNESS_REPORT.md`).
* Replay is deterministic: `test_replay_is_deterministic` (API) and `test_decision_is_deterministic_and_hash_stable` (engine).

## 5. The 17 known defects

| # | Defect | Fix | Regression test |
|---|---|---|---|
| 1 | Cross-application upload overwrite / RAG leakage | Per-application storage, magic-byte + size checks, per-application index | `test_uploads_are_isolated_per_application`, `test_store_upload_isolates_per_application` |
| 2 | Stored XSS in dashboard | New UI renders only via `x-text`/`textContent`, CSP | `test_frontend_never_uses_innerhtml` |
| 3 | Pipeline ignored missing documents | Document gate → `BELGE_BEKLENIYOR` + letter, auto-resume | `test_missing_documents_stop_the_flow_with_a_letter` |
| 4 | "DTI" was debt stock / income | DSR = monthly instalments / net income | `test_high_dsr_gets_conditional_counter_offer_with_counterfactual`, `test_annuity_and_with_loan` |
| 5 | Suggested amount = income×6 regardless | Limit engine, counter-offer only when approvable; counterfactual on decline | `test_clean_applicant_gets_offer`, `test_high_dsr_…` |
| 6 | Mock LLM text, broken chain, no timeout/retry/base_url, temperature ignored | LLM contract (ADR 0006) | `tests/test_llm.py` (26 tests) |
| 7 | PII in prompts and logs | Redactor + PII log filter | `test_live_requests_contain_no_pii`, `test_log_filter_masks_pii` |
| 8 | Circuit breaker recreated per run | Process-wide registry, Redis-shared state | `test_breaker_is_shared_across_client_instances`, `test_redis_store_shares_state_between_breakers` |
| 9 | Worker state invisible to API, lost on restart | Single database for API and worker | `test_db_metrics_llm_status_and_audit_chain` |
| 10 | Listing sorted by random id | `created_at DESC` | `test_listing_is_newest_first` |
| 11 | Broken Turkish glyphs in PDF | Embedded DejaVu Sans | `test_report_pdf_has_turkish_glyphs` (checks `ığşİ`) |
| 12 | Dashboard links used server file paths | Authenticated `/report` and `/contract` endpoints | `test_report_pdf_has_turkish_glyphs`, `test_offer_acceptance_produces_contract` |
| 13 | Outbox rewrote whole file, never sent | Transactional outbox + adapters, idempotency | `test_notifications_outbox`, `test_outbox_enqueue_is_idempotent` |
| 14 | Missing PyYAML, unused deps, timeout not passed, dead config | Requirements cleaned, timeout wired to httpx, config pruned | `test_timeout_setting_reaches_httpx`, `test_dockerfile_contract` |
| 15 | Unused duplicate-TCKN check, no checksum | Velocity rule via blind index, TCKN checksum | `test_invalid_tckn`, `test_submission_validation[checksum]` |
| 16 | No auth/RBAC/CORS/rate limit; inline mode blocked requests | JWT/RBAC, ownership, slowapi, CORS whitelist; background dispatch | `test_endpoints_require_authentication`, `test_role_restrictions`, `test_applicant_cannot_see_others` |
| 17 | Hard-coded rate, weights, employers | YAML rules (`rules/*.yaml`), risk-based pricing | `test_policy_file_parses_and_is_versioned`, `tests/test_pricing.py` |

## 6. Quality gates
`ruff check .` ✓ · `ruff format --check .` ✓ · `mypy app` ✓ (98 files, no issues) · `pytest --cov=app` → **231 passed, 89.3 % coverage** (was 66 tests) · `docker compose build` ✓. The CI workflow steps were additionally executed inside the Linux image (`python:3.11-slim`) with identical results.

## 7. Documentation and CI
README (positioning, Mermaid architecture + state machine, comparison table, screenshots, 30-second start, demo users and personas, LLM modes, metrics), `docs/ARCHITECTURE.md`, seven ADRs, model card, fairness report, compliance notes, performance report, demo script, changelog, MIT licence and `.github/workflows/ci.yml`.

## 8. Git rules
Small Conventional Commits pushed to `origin/main`; no commit carries AI attribution; `.env` and the task file are not tracked (`.gitignore`, `.git/info/exclude`); staged diffs were secret-scanned before every commit (0 hits). Release tag `v2.0.0`.

## Known limitations / follow-ups
* The legacy `src/` folder (untracked, only `__pycache__`) could not be deleted from this session because of a local permission rule; it is ignored by git. Delete it with `rm -rf src`.
* Optional extras (OCR, barcode decoding, docling, dice-ml, EBM, evidently, sentence-transformers, lifelines) degrade gracefully when absent; they are not installed in the default image.
* GitHub Actions status could not be read from this environment (private repository, no `gh` CLI); the identical steps pass locally and in a Linux container.
