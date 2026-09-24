# ADR 0007 — Dependency-free web UI (Alpine.js + Chart.js, vendored)

**Status:** accepted

## Context
The UI must be demo-ready, Turkish, responsive and dark/light, but the project is backend-centric and must run offline with a single `docker compose up`.

## Decision
A single-page app served by FastAPI from `app/static` using vendored Alpine.js and Chart.js (no Node build, no CDN at runtime). User data is rendered exclusively through `x-text`/`textContent` (a test forbids `innerHTML`), fixing the previous stored-XSS issue. A strict CSP is sent on every response.

## Consequences
+ Zero build step, tiny image, no supply-chain surface from npm.
− Less component reuse than React/shadcn; acceptable for three views.
