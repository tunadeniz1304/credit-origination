# Performance and Resilience

Measured with `scripts/load_test.py` against `docker compose up` on a developer laptop (Windows 11, Docker Desktop, 8 GB VM): 4 gunicorn/uvicorn API workers, one Celery worker with `--concurrency=2`, PostgreSQL 16, Redis 7, LLM in demo mode (to avoid 300 paid API calls). Each application uploads five generated PDFs (identity, payslip, SGK record, e-Devlet residence document, bank statement) that go through extraction and tamper analysis synchronously in the upload request.

## Throughput run — 100 concurrent applicants (20 client threads)

| Metric | Value |
|---|---|
| Applications completed | **100 / 100** |
| Wall time | 104 s (0.96 applications/s end to end, incl. 500 document uploads) |
| End-to-end latency p50 / p95 / max | 15.9 s / 38.7 s / 43.4 s |
| Outcomes | 46 offers, 46 specialist reviews, 8 automatic rejections |

End-to-end latency is dominated by the synchronous document analysis in the upload requests and the two Celery worker processes; the decision stage itself (rules + PD + SHAP + scorecard + pricing + counterfactuals) takes 40–250 ms (`decisions.latency_ms`, visible on the dashboard). Scaling knobs: `WEB_CONCURRENCY`, Celery `--concurrency`, more worker replicas.

Findings fixed during the test:
* The upload endpoint was `async def` doing blocking PDF work, which starved the event loop until gunicorn killed the workers (`WORKER TIMEOUT`). It now runs in the threadpool.
* The audit-chain lock (PostgreSQL advisory transaction lock) was held while narratives were generated; slow work now happens before the first audit append.
* Models are warmed at start-up so the first request does not pay for model + SHAP initialisation.

## Resilience run — KKB fault injection

With `FAULT_INJECTION_RATE=0.3` the Tenacity retries (3 attempts) absorb almost all failures. With `FAULT_INJECTION_RATE=0.95`:

| Observation | Value |
|---|---|
| KKB circuit breaker transitions to OPEN (shared state in Redis) | 6 |
| Applications parked in `VERI_TOPLANIYOR` (no failure, no data loss) | 10 / 10 |
| `DATA_COLLECTION_RETRY` audit entries | 30 |
| Time to drain after the fault was removed (beat `retry_stalled` every 60 s + Celery countdown retries) | ≈ 40 s, 10 / 10 completed |

## Reproduce

```bash
docker compose up -d --build
python scripts/load_test.py --count 100 --concurrency 20
# resilience: restart the worker with FAULT_INJECTION_RATE=0.95, run a few applications, then set it back to 0
```
