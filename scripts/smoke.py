"""End-to-end API smoke test (runnable against a live server).

Walks the complete surface: health -> submit (approved + rejected vectors)
-> fetch -> schedule -> scorecard -> offer -> audit -> notifications ->
dispatch -> metrics -> queue -> dashboard. Prints a PASS/FAIL summary and
exits non-zero when any step fails.

Usage:
    python scripts/smoke.py [--base http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

APPROVE_IDENTITY = "12345678901"
REJECT_IDENTITY = "34567890123"
ALL_DOCS = ["IDENTITY", "INCOME", "EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"]


def request(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc


def submit(base: str, identity: str, income: float, amount: float, term: int) -> str:
    status, body = request(
        f"{base}/api/v1/applications",
        "POST",
        {
            "name": "Smoke",
            "identity_no": identity,
            "monthly_income": income,
            "requested_amount": amount,
            "requested_term_months": term,
            "submitted_documents": ALL_DOCS,
        },
    )
    assert status == 202, f"submit status {status}"
    return body["application_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL':4} {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    status, body = request(f"{base}/health")
    check("health", status == 200 and body["status"] == "ok", f"backend={body.get('queue_backend')}")

    up = submit(base, APPROVE_IDENTITY, 300_000, 100_000, 36)
    status, body = request(f"{base}/api/v1/applications/{up}")
    check("approved submit+fetch", status == 200 and body["status"] == "APPROVED")

    for path in ("schedule", "scorecard", "offer", "audit"):
        s, _ = request(f"{base}/api/v1/applications/{up}/{path}")
        check(f"approved.{path}", s == 200, f"status={s}")

    status, body = request(f"{base}/api/v1/applications/{up}/report")
    check("report.json download", status == 200 and body.get("status") == "APPROVED")

    rej = submit(base, REJECT_IDENTITY, 30_000, 50_000, 24)
    status, body = request(f"{base}/api/v1/applications/{rej}")
    check("rejected submit", status == 200 and body["status"] == "REJECTED")

    s, _ = request(f"{base}/api/v1/applications/{rej}/schedule")
    check("rejected.schedule is 409", s == 409, f"status={s}")

    status, body = request(f"{base}/api/v1/notifications")
    check("notifications list", status == 200 and body["pending"] >= 1)
    nid = body["entries"][0]["id"]
    status, body = request(f"{base}/api/v1/notifications/{nid}/deliver", "POST")
    check("notification deliver", status == 200 and body["delivered"] is True)

    status, body = request(f"{base}/api/v1/notifications/dispatch", "POST")
    check("outbox dispatch", status == 200)

    status, body = request(f"{base}/api/v1/metrics")
    check("metrics", status == 200 and body["total_applications"] >= 2)

    status, body = request(f"{base}/api/v1/queue")
    check("queue status", status == 200 and "app.tasks.process_application" in body["registered_tasks"])

    status, _ = request(f"{base}/static/dashboard.html")
    check("dashboard served", status == 200)

    if failures:
        print(f"\nSMOKE FAILED: {len(failures)} failures -> {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nSMOKE PASS: all steps OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
