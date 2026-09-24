# ADR 0007 — Dependency-free web UI (Alpine.js + Chart.js, vendored)

**Status:** accepted

## Context
The UI must be demo-ready, Turkish, responsive and dark/light, but the project is backend-centric and must run offline with a single `docker compose up`.

## Decision
A single-page app served by FastAPI from `app/static` using vendored Alpine.js and Chart.js (no Node build, no CDN at runtime). User data is rendered exclusively through `x-text`/`textContent` (a test forbids `innerHTML`), fixing the previous stored-XSS issue. A strict CSP is sent on every response.

Update (v2): the UI uses Alpine's CSP build (`@alpinejs/csp` 3.17.4, vendored as `alpine-csp-3.17.4.min.js`), which evaluates directive expressions with its own parser instead of `new Function`. The CSP is therefore `script-src 'self'` with neither `'unsafe-eval'` nor `'unsafe-inline'`; templates keep to simple expressions and anything more complex is a component method in `app.js`.

## Consequences
+ Zero build step, tiny image, no supply-chain surface from npm.
− Less component reuse than React/shadcn; acceptable for three views.
