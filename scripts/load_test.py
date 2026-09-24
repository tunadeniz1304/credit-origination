"""Load and resilience test: N concurrent applications against a running server.

Submits ``--count`` applications with ``--concurrency`` threads (documents
included), waits until each reaches a terminal decision state and reports
end-to-end p50/p95 latencies plus the per-stage p95 from the Prometheus
histograms. Run the server with ``FAULT_INJECTION_RATE=0.3`` to exercise the
KKB circuit breaker: affected applications stay in ``VERI_TOPLANIYOR`` and are
retried by the worker until they complete.

Usage: python scripts/load_test.py [--base URL] [--count 100] [--concurrency 20]
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.documents.samples import generate_applicant_bundle
from app.integrations import personas
from app.kyc.tckn import synthetic_tckn
from scripts.smoke import Client

DONE = {"TEKLIF_SUNULDU", "UZMAN_INCELEMESI", "OTOMATIK_RET"}


def one(base: str, index: int, tmp: Path, timeout: float) -> dict:
    c = Client(base)
    c.login("basvuran")
    tckn = synthetic_tckn(f"load-{index}-{time.time()}")
    income = 30_000 + (index % 7) * 5_000
    body = {
        "name": f"Yük Testi {index}",
        "identity_no": tckn,
        "phone": f"0544{index:07d}",
        "email": "load@example.com",
        "address": f"Yük Sok. No:{index}",
        "iban": f"TR44000610000000{index:010d}",
        "monthly_income": income,
        "requested_amount": income * 4,
        "requested_term_months": 36,
        "consents": {
            "kvkk_aydinlatma": True,
            "acik_riza": True,
            "kkb_sorgu": True,
            "edevlet_sorgu": True,
            "acik_bankacilik": True,
        },
    }
    started = time.perf_counter()
    status, created = c.call("/api/v1/applications", "POST", body)
    if status != 202:
        return {"ok": False, "error": status}
    app_id = created["application_id"]
    ob = personas.open_banking_transactions(tckn, income)
    files = generate_applicant_bundle(
        tmp / str(index),
        name=body["name"],
        tckn=tckn,
        iban=body["iban"],
        address=body["address"],
        employer="Yük A.Ş.",
        net_income=income,
        transactions=ob["transactions"],
        opening_balance=ob["account"]["opening_balance"],
    )
    for code, path in files.items():
        c.upload(app_id, code, path)
    deadline = time.time() + timeout
    state = ""
    while time.time() < deadline:
        _, detail = c.call(f"/api/v1/applications/{app_id}")
        state = detail["state"]
        if state in DONE:
            break
        time.sleep(1.0)
    return {
        "ok": state in DONE,
        "state": state,
        "seconds": time.perf_counter() - started,
        "retries": detail.get("retry_count", 0),
    }


def stage_p95(metrics: str) -> dict[str, float]:
    """Approximate p95 per stage from cumulative histogram buckets."""
    out: dict[str, float] = {}
    buckets: dict[str, list[tuple[float, float]]] = {}
    for line in metrics.splitlines():
        m = re.match(
            r'anil2_stage_duration_seconds_bucket\{le="([^"]+)",stage="([^"]+)"\} ([0-9.e+]+)', line
        )
        if m:
            le = float("inf") if m.group(1) == "+Inf" else float(m.group(1))
            buckets.setdefault(m.group(2), []).append((le, float(m.group(3))))
    for stage, rows in buckets.items():
        rows.sort()
        total = rows[-1][1]
        out[stage] = (
            next((le for le, count in rows if count >= 0.95 * total), float("inf"))
            if total
            else 0.0
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="anil2-load-"))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: one(args.base, i, tmp, args.timeout), range(args.count)))
    wall = time.perf_counter() - started
    done = [r for r in results if r["ok"]]
    seconds = sorted(r["seconds"] for r in done)
    states: dict[str, int] = {}
    for r in results:
        states[r.get("state", "ERROR")] = states.get(r.get("state", "ERROR"), 0) + 1
    print(
        f"applications={args.count} completed={len(done)} wall={wall:.1f}s throughput={len(done) / wall:.2f}/s"
    )
    if seconds:
        p95 = seconds[max(0, int(len(seconds) * 0.95) - 1)]
        print(
            f"end-to-end p50={statistics.median(seconds):.1f}s p95={p95:.1f}s max={seconds[-1]:.1f}s"
        )
    print("states:", states, "retried:", sum(1 for r in done if r.get("retries")))
    _, metrics = Client(args.base).call("/metrics")
    text = metrics.decode() if isinstance(metrics, bytes) else str(metrics)
    print("stage p95 (s):", stage_p95(text))
    return 0 if len(done) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
