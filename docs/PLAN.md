# Transformation Plan — Anil2 v2

This plan turns the "Smart Credit Operations Agent" prototype into an end-to-end,
explainable, human-in-the-loop credit origination platform. Work proceeds in
phases F0–F9; every phase ends with the quality gate
(`ruff check`, `ruff format --check`, `mypy app`, `pytest --cov=app`, `docker compose build`).

## Verified starting point (F0 discovery)

- Python 3.11, FastAPI, pydantic v2, Celery/Redis with an inline fallback, 66 green tests.
- State lives in JSON files plus an in-memory `records` dict (breaks across processes).
- Only four hard committee rules; the "DTI" is debt stock / monthly income (wrong concept).
- The mock LLM leaks `[mock-llm]` text into reports; prompts carry name + TCKN.
- Uploads are stored globally per document code (cross-application overwrite and leakage).
- No auth, no rate limit, no CORS policy; dashboard renders names through `innerHTML`.
- The 17 known defects listed in the brief were all reproduced by reading the code.

## Target architecture

```
app/
  core/          settings, logging (PII filter + request id), security (JWT/RBAC),
                 crypto (Fernet + HMAC blind index), YAML rule loading, metrics
  db/            SQLAlchemy 2.0 models, session factory, repositories (Alembic migrations)
  workflow/      application state machine + orchestration service + pipeline stages
  kyc/           TCKN checksum, sanctions/PEP fuzzy screening, velocity, fraud-ring graph,
                 IsolationForest anomaly score
  documents/     isolated storage, text extraction/OCR fallback, field extraction,
                 fraud signals (pikepdf metadata, fonts, arithmetic), reconciliation
  integrations/  KKB/Findeks, e-Devlet/SGK, GİB, open-banking mocks; shared circuit breaker
  cashflow/      transaction categorisation + 20+ cash-flow features
  decisioning/   YAML policy DSL (safe evaluator), monotonic LightGBM PD, WoE scorecard,
                 decision policy, limit engine, replay
  explainability/ SHAP reason codes (Turkish), counterfactual search, letters
  pricing/       PD×LGD×EAD, RAROC solver, BSMV/KKDF, legal cap, APR, schedule
  workbench/     queue + SLA, authority matrix, overrides, four-eyes, KVKK m.11 objections
  agents/        LLM providers (OpenAI-compatible / Anthropic / deterministic), redaction,
                 narrator with citation guard, LangGraph underwriter agent, policy RAG
  governance/    fairness (AIR, fairlearn), drift (PSI), model inventory + cards,
                 champion/challenger, rule-set backtests
  ews/           early-warning model + watchlist
  static/        dependency-free SPA (Alpine.js + Chart.js vendored)
```

## Key decisions (see `docs/adr/`)

1. Hybrid decisioning: knock-out policy rules → calibrated monotonic LightGBM PD →
   transparent WoE scorecard; the LLM never decides.
2. PostgreSQL in Docker, SQLite locally/tests; Alembic migrations; transactional outbox.
3. PII encrypted at the application layer (Fernet) with an HMAC blind index for TCKN.
4. OpenAI-compatible LLM client so the model can be moved on-prem (vLLM/Ollama) for
   BDDK data-localisation; deterministic demo mode when no key is configured.
5. Frontend: dependency-free Alpine.js + Chart.js SPA (vendored, no Node build step).

## Phase checklist

| Phase | Scope | Exit criterion |
|---|---|---|
| F0 | Discovery, tooling (`pyproject.toml`, `Makefile`), PyYAML, this plan | 66 tests green |
| F1 | LLM contract, 17 bug fixes, security foundation | LLM + regression tests |
| F2 | Persistence + state machine + outbox, integrations + consent | transition + parity tests |
| F3 | KYC/fraud, IDP + tamper detection, cash-flow analytics | tampered-doc + feature tests |
| F4 | PD model, scorecard, decision policy, explainability, pricing | metrics, replay, reason codes |
| F5 | Workbench, objections, observability | authority matrix / four-eyes tests |
| F6 | Underwriter agent + policy RAG | fake-LLM tool use + citation tests |
| F7 | Governance + EWS | fairness, drift, champion/challenger tests |
| F8 | UI, demo personas, load test | E2E smoke |
| F9 | Docs, CI, polish, release tag | all quality gates |
