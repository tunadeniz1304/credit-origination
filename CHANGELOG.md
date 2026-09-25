# Changelog

## [2.1.0] — 2026-09-25

Closes the findings of the independent v2 audit (7.5/10) and validates the modelling methodology on real public credit data. Details: `docs/FINAL_REPORT_v2.md`.

### Added
- Real public data: `scripts/fetch_public_credit_data.py` (UCI Taiwan, German Credit, pinned SHA-256, CC BY 4.0 attribution), 3,000-row offline fixture, `docs/DATA.md`.
- Lane A validation (`app/validation/`, `scripts/run_validation.py`, `docs/VALIDATION_REPORT.md`): stratified 5-fold CV + 20 % hold-out, bootstrap and DeLong AUC CIs, own DeLong test (checked on a hand-worked example), decile calibration, ECE, Hosmer–Lemeshow, evidence-based champion rule, real-attribute fairness at one approval rate, LDA search (proxy weakening, ExponentiatedGradient, group thresholds with legal caveat).
- Lane B anchoring (`app/decisioning/public_mapping.py`, `scripts/run_lane_b.py`): public→bureau feature mapping, real-data `bureau_behavior_score`, generator anchored to the real default curve per delinquency band, PD level and low-risk calibration checks. Models retrained as v2.
- `GET /api/v1/models/validation`, real-data evidence in champion/challenger and fairness endpoints, "Model doğrulama" UI tab.
- Turkish label dictionary (`rules/labels.yaml`, `GET /api/v1/labels`), OCR status in `/health`, Tesseract in the image.
- Real-server concurrency test (20 applications + 20 logins), pyinstrument profiling script, Playwright E2E tests.

### Changed
- Policy recalibrated to the anchored PD scale (`policy_v2`, `authority_v2`); pricing uses the Basel other-retail correlation (`pricing_v2`); reason codes need materiality and an adverse value range (`reasons_v2`).
- Promotion of a challenger is refused when real-data evidence shows it is significantly weaker.
- Browser sessions use HttpOnly cookies with signed double-submit CSRF; demo users are off in prod by default; registration can be disabled, is rate limited and has a CAPTCHA hook.
- CSP without `unsafe-eval` (Alpine CSP build); accessible UI (labels, keyboard, WCAG AA contrast, chart tables); paginated, searchable queue.
- README: "Inspired-by patterns" and "Limitations" replace the vendor comparison; regulatory citations verified with sources.

### Fixed
- DSR re-checked with the priced, taxed instalment (limit ↔ price iteration).
- Meaningless reason codes (52-month tenure as "short"); raw enum codes in PDFs.
- Producer-dependent fraud detection; OCR path requiring the Tesseract binary.
- SQLite "database is locked" 500s (single writer lock, short transactions, savepoint retry for the audit chain).
- Non-atomic circuit breaker with unlimited HALF_OPEN probes (Lua scripts); duplicate pending reviews (partial unique index + migration 0002).
- KVKK redaction missing Turkish İ/ı and ASCII-folded name variants; internal LLM host in tracked files.

## [2.0.0] — 2026-09-24

Transformation of the "Smart Credit Operations Agent" prototype into an end-to-end, explainable credit origination platform.

### Added
- OpenAI-compatible LLM contract (live / demo / fallback), PII redaction, citation-and-number guard, three-step narrative chain, `/api/v1/llm/status`, `scripts/llm_smoke.py`.
- PostgreSQL/SQLite persistence with Alembic, application state machine, hash-chained audit log, transactional outbox with webhook/SMTP adapters.
- JWT + RBAC (six roles), ownership checks, rate limiting, CORS whitelist, CSP/security headers, request ids, Fernet PII encryption with blind indexes, KVKK retention job.
- KYC: TCKN checksum, sanctions/PEP fuzzy screening, velocity, fraud-ring graph, IsolationForest anomaly score.
- Intelligent document processing: magic-byte upload validation, field extraction with confidence/bbox, tamper signals (incremental saves, producer, fonts, arithmetic, e-Devlet barcode), income reconciliation, synthetic document generator.
- Persona-consistent KKB/Findeks, SGK, GİB and open-banking mocks behind consent gates, retries and a shared circuit breaker.
- 24 cash-flow features; hybrid decision engine (YAML policy DSL, monotone LightGBM PD with isotonic calibration, WoE scorecard, shadow challenger), limits and counter-offers, replay endpoint.
- SHAP-based Turkish reason codes, counterfactual explanations, applicant letters with the KVKK art. 11 objection right.
- Risk-based pricing (PD×LGD×EAD, Basel IRB capital, RAROC-solved rate, BSMV/KKDF, TCMB cap, APR).
- Underwriter workbench (SLA queue, authority matrix, four-eyes, overrides, field corrections, objections, disbursal).
- Supervised LangGraph underwriter agent with cited memoranda; BM25 policy/regulation assistant.
- Governance: model inventory and cards, fairness (AIR, less-discriminatory alternatives), PSI drift, champion/challenger promotion, rule-set backtests; early-warning watchlist.
- Web UI (applicant portal, workbench, management dashboard), six demo scenarios, HTTP smoke and load tests, Prometheus metrics, health probes, CI.

### Fixed
All 17 defects listed in the transformation brief (cross-application upload leakage, stored XSS, missing document gate, DSR definition, suggested amount, placeholder LLM text, PII in prompts/logs, per-request circuit breakers, stale worker state, ordering, Turkish PDF glyphs, broken report links, outbox rewrite, missing dependencies, unused duplicate check, missing auth, hard-coded rates).

### Removed
JSON-file result store, in-memory records and the legacy dashboard.
