# Performance and Resilience

## v2.1 — inline mode (SQLite), before / after

Same laptop, `scripts/load_test.py`, LLM in demo mode, `TASK_QUEUE_BACKEND=inline`, a fresh SQLite
database per run (`loadrun` helper: one uvicorn process, five generated PDFs per application).
"Before" is tag `v2.0.0`, "after" is `main` at v2.1.0.

| Scenario | v2.0.0 | v2.1.0 |
|---|---|---|
| One clean application at a time (10 sequential), end-to-end p50 / p95 | 1.6 s / 1.7 s | **1.6 s / 1.7 s** |
| 40 applications, 20 concurrent clients | did **not finish** in 12 min (SQLite writer deadlock; client stopped) | **40 / 40** in 35 s, p50 16.1 s / p95 16.8 s, 0 retries |

* **Target (p50 < 5 s for a clean application in inline mode) is met.** The 15.9 s figure in the v1
  audit was a p50 *under load* (100 applicants, 20 threads, Docker); it measured queueing, not the
  pipeline. Single-application latency was already ≈ 1.6 s in v2.0.0 and is unchanged.
* The real v2.1 change is **correctness under concurrency** in inline mode: the pipeline now prepares
  the decision read-only and writes in one short transaction, and a process-wide SQLite writer lock is
  taken at the start of every write transaction (avoids `SQLITE_BUSY_SNAPSHOT` upgrade deadlocks);
  audit appends retry inside a savepoint. Under 20 concurrent clients the latency is queueing behind the
  single SQLite writer — use the Docker topology (PostgreSQL + Celery) for throughput.
* `tests/test_concurrency.py` runs 20 applications (five uploads each) + 20 logins in parallel against a live
  server and asserts 0 HTTP 500s and a valid audit chain.

Reproduce (inline):

```bash
TASK_QUEUE_BACKEND=inline LLM_MODE=demo uvicorn app.main:app --port 8634 &
python scripts/load_test.py --base http://127.0.0.1:8634 --count 10 --concurrency 1
python scripts/load_test.py --base http://127.0.0.1:8634 --count 40 --concurrency 20
```

## v1 — Docker topology

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
