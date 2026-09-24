# Architecture

## Runtime topology

| Service | Role |
|---|---|
| `app` | FastAPI (gunicorn + uvicorn workers). Runs `alembic upgrade head` on start, serves the API and the static UI. |
| `worker` | Celery worker executing the staged pipeline, outbox dispatch, retries, retention and drift jobs. |
| `beat` | Celery beat schedule: outbox every 15 s, stalled-application retry every 60 s, KVKK retention daily, drift hourly. |
| `postgres` | System of record (applications, decisions, audit chain, outbox, models, rule sets …). |
| `redis` | Broker/result backend and cross-process circuit-breaker state. |
| `mailhog` | Development SMTP sink for applicant letters. |

Without Docker the same code runs with SQLite and the inline dispatcher (FastAPI background tasks), so the test suite needs no external service.

## Packages

| Package | Responsibility |
|---|---|
| `app/core` | Settings (LLM contract, security, persistence), YAML rule loaders, logging with PII masking and request ids, JWT/RBAC, Fernet + blind index, Prometheus metrics, rate limiter |
| `app/db` | SQLAlchemy 2.0 models, sessions, hash-chained audit log, transactional outbox |
| `app/workflow` | State machine, `ApplicationService` (transitions + audit + outbox), staged `Pipeline`, offers/contract/disbursal, dispatch |
| `app/kyc` | TCKN checksum, sanctions/PEP screening, fraud-ring graph, anomaly score |
| `app/documents` | Upload storage, extraction (PyMuPDF, optional OCR), tamper signals, synthetic document generator |
| `app/integrations` | Persona-consistent mocks for KKB, SGK, GİB, open banking; consent gate; retries + shared circuit breaker |
| `app/cashflow` | Transaction categorisation and 24 cash-flow features |
| `app/decisioning` | Feature snapshot, safe DSL, rules engine, model registry, training, hybrid engine |
| `app/explainability` | SHAP reason codes, counterfactual search |
| `app/pricing` | RAROC pricing, taxes, legal cap, APR, taxed schedule |
| `app/workbench` | Queue/SLA, authority matrix, maker-checker, objections |
| `app/agents` | LLM providers, service (modes/fallback/telemetry), redaction, citation guard, narrator, RAG, LangGraph underwriter agent, policy assistant |
| `app/governance` | Fairness, drift, model inventory/cards, champion/challenger, rule-set governance |
| `app/ews` | Early-warning model and watchlist |
| `app/reports` | Credit memorandum PDF/JSON, AI memo PDF (DejaVu Sans for Turkish) |
| `app/api` | Routers, dependencies, views (role-aware projections, masking) |

## Request/decision flow

1. `POST /api/v1/applications` validates the TCKN checksum, phone, IBAN and mandatory consents, encrypts PII, records consents and moves `TASLAK → GONDERILDI`, then dispatches processing.
2. **Document gate** — missing documents move the application to `BELGE_BEKLENIYOR` with a generated letter; each upload is type-checked by magic bytes, stored per application, extracted and scored for tampering. Completing the set resumes the pipeline.
3. **Data collection** — KKB/SGK (+GİB) and open banking are called only with valid consents, through Tenacity retries and a shared circuit breaker. Outages keep the application in `VERI_TOPLANIYOR` and it is retried. KYC (sanctions, velocity, ring, anomaly) and income reconciliation run here.
4. **Decision** — the snapshot is scored by the hybrid engine; the decision row stores rule-set/model versions and the snapshot hash. Narratives (live/demo/fallback), the PDF report and the outbox letter follow; approvals get a priced offer.
5. **Workbench** — referrals are queued with SLA and priority; specialist decisions are checked against the authority matrix and four-eyes rules.
6. **Offer → contract → disbursal** — acceptance creates the pre-contract information form; disbursal seeds the early-warning history.

## Concurrency and consistency

* Every stage commits; API and worker read the same database (fixes the in-memory/JSON inconsistency).
* The audit chain is serialised with a PostgreSQL advisory transaction lock (SQLite: a process lock released when the root transaction ends); slow work (LLM narratives) runs before the first append to keep the lock short.
* LLM call telemetry is buffered and flushed after commit to avoid write contention.
