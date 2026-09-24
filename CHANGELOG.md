# Changelog

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
