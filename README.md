# Akıllı Kredi Operasyon Ajanı

Enterprise-scale **Smart Credit Operations Agent**: an async-queue-based credit
application processing platform. A FastAPI backend accepts applications and
enqueues them for background processing; agents then validate documents
(RAG-powered), collect financial data from external providers (circuit
breaker + retry), and run a prompt-chained credit decision engine that emits a
BDDK-style **Kredi Tahsis Raporu** (credit allocation report) as both JSON and
PDF.

> Turkish-facing outputs: missing-document request drafts and the credit
> committee report are produced in Turkish. Code, comments and docs are in English.

## Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                 FastAPI (app/)               │
                        │  POST /api/v1/applications → enqueue task    │
                        │  GET  /api/v1/applications/{id} → store/Redis│
                        └──────────────────────┬───────────────────────┘
                                               │ TaskDispatcher
                        ┌──────────────────────┴───────────────────────┐
                        │         celery | inline (no broker)          │
                        └──────────────────────┬───────────────────────┘
                                               ▼
        ┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
        │  Document / RAG agent    │   Integration agent      │   Decision engine        │
        │  LangChain loaders       │   KKB (mock) client      │   LLM prompt chain       │
        │  chunk + retrieve        │   e-Devlet(mock) client  │   deterministic factors  │
        │  missing-doc draft (TR)  │   CircuitBreaker+Tenacity│   Kredi Tahsis Raporu    │
        └──────────────────────────┴──────────────────────────┴──────┬───────────────────┘
                                                                     ▼
                                           data/reports/*.{json,pdf}   data/results/*
```

- `app/models/` — Pydantic domain models (single source of truth).
- `app/core/` — settings, business rules, logging, worker-agnostic `TaskDispatcher`.
- `app/agents/` — RAG document agent, `LLMProvider` abstraction, decision chain.
- `app/integrations/` — external API clients with circuit breaker + retry.
- `app/api/` — FastAPI router.
- `app/worker/` — Celery app and task registration (plus inline equivalents).
- `app/engine/` — pipeline orchestration, persistence store and BDDK report generation.
- `config/config.json` — business rules (committee thresholds, mandatory documents).

### Queue modes

`TASK_QUEUE_BACKEND` (or `.env`) selects the execution backend:

| Value      | Behaviour                                                        |
|------------|------------------------------------------------------------------|
| `auto`     | Ping `REDIS_URL`; Celery if reachable, else inline (default)     |
| `celery`   | Enqueue on the Redis broker (used in Docker compose)             |
| `inline`   | Run the identical task body synchronously — no broker required   |

Each domain task is implemented once and registered both as a Celery task and
as an inline callable, so local development and the test suite run the exact
same code a worker would run.

### LLM provider

`app/agents/llm.py` exposes `LLMProvider.complete(system, user)`:

- `OpenAIProvider` — `openai` SDK, active when `OPENAI_API_KEY` is set.
- `AnthropicProvider` — `anthropic` SDK, active when `ANTHROPIC_API_KEY` is set.
- `MockLLMProvider` — deterministic, offline-safe fallback (default).

### Deterministic external integrations

KKB and e-Devlet values derive from a stable SHA-256 digest of the identity
number (never Python's salted `hash()`), so results are reproducible across
processes. Verified demo identity vectors:

| Identity       | KKB score | Total debt | Outcome                 |
|----------------|-----------|------------|-------------------------|
| `12345678901`  | 1450      | 134 070    | passes `min_kbb_score`  |
| `34567890123`  | 619       | 176 116    | fails `min_kbb_score`   |

## Quickstart (local, no Docker)

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env            # TASK_QUEUE_BACKEND=inline by default
uvicorn app.main:app --reload
```

Smoke:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/v1/ping      # {"backend":"inline", ...}

# Submit an application (returns an APP-* id; inline mode finalises inline).
curl -X POST http://127.0.0.1:8000/api/v1/applications \
  -H "Content-Type: application/json" \
  -d '{"name":"Ali Yılmaz","identity_no":"12345678901","monthly_income":300000,
       "requested_amount":100000,"requested_term_months":36,
       "submitted_documents":["IDENTITY","INCOME","EMPLOYMENT","ADDRESS","BANK_STATEMENT"]}'

# Poll the result; approved applications expose data/reports/{id}_report.{json,pdf}.
curl http://127.0.0.1:8000/api/v1/applications/APP-XXXXXXXXXXXXXXXX
```

## Tests

```bash
python -m pytest -q
```

Unit tests exercise the pipeline through the inline dispatcher and mocked
HTTP (respx) — no Redis or live network required.

## Docker

```bash
docker compose up --build
```

Runs three services: `app` (FastAPI + uvicorn on port 8000), `worker`
(Celery on the Redis broker; command overridden to start the worker), and
`redis`. `TASK_QUEUE_BACKEND=celery` is forced via environment and both app
and worker wait for a healthy Redis before starting.

## API

| Method | Path                                    | Purpose                                   |
|--------|-----------------------------------------|-------------------------------------------|
| GET    | `/`                                     | Operations dashboard (static web UI)      |
| GET    | `/health`                               | Liveness + queue backend                  |
| POST   | `/api/v1/ping`                          | Queue round-trip smoke test               |
| POST   | `/api/v1/applications`                  | Submit an application                     |
| GET    | `/api/v1/applications`                  | List applications (newest first)          |
| GET    | `/api/v1/applications/{id}`             | Poll status + generated report            |
| POST   | `/api/v1/applications/{id}/documents`   | Deliver a missing document (multipart)    |
| GET    | `/api/v1/applications/{id}/documents`   | Document-control snapshot                 |
| POST   | `/api/v1/applications/{id}/reprocess`   | Re-enqueue after document delivery        |
| GET    | `/api/v1/applications/{id}/schedule`    | Amortization (repayment) plan             |
| GET    | `/api/v1/applications/{id}/scorecard`   | Composite BDDK-style risk score + grade   |
| GET    | `/api/v1/applications/{id}/offer`       | Priced loan offer terms (approved only)   |
| GET    | `/api/v1/applications/{id}/report`      | Download JSON/PDF report file             |
| GET    | `/api/v1/applications/{id}/audit`       | Append-only lifecycle audit trail         |
| GET    | `/api/v1/metrics`                       | Status distribution + suggested-volume    |
