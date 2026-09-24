"""End-to-end smoke test against a running server (Docker or local).

Walks the full journey through HTTP only: login → application → document
upload → decision → offer → acceptance → disbursal, plus a grey-zone review
with four-eyes, an automatic rejection with a KVKK objection, the AI memo,
governance endpoints and the dashboard. Exit code 1 on any failure.

Usage: python scripts/smoke.py [--base http://127.0.0.1:8000] [--timeout 120]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.documents.samples import generate_applicant_bundle
from app.integrations.personas import DEMO_TCKN, open_banking_transactions

PASS = "Demo123!"


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.token: str | None = None

    def call(
        self,
        path: str,
        method: str = "GET",
        payload: dict | None = None,
        raw: bytes | None = None,
        ctype: str | None = None,
    ):
        headers = {}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        if raw is not None:
            data, headers["Content-Type"] = raw, ctype or "application/octet-stream"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                try:
                    return resp.status, json.loads(body)
                except ValueError:
                    return resp.status, body
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()[:300]

    def login(self, user: str) -> None:
        self.token = None
        status, body = self.call("/api/v1/auth/login", "POST", {"username": user, "password": PASS})
        assert status == 200, body
        self.token = body["access_token"]

    def upload(self, app_id: str, code: str, path: Path):
        boundary = uuid.uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="code"\r\n\r\n{code}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                "Content-Type: application/pdf\r\n\r\n"
            ).encode()
            + path.read_bytes()
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self.call(
            f"/api/v1/applications/{app_id}/documents",
            "POST",
            raw=body,
            ctype=f"multipart/form-data; boundary={boundary}",
        )


def application(persona: str, suffix: str, income: float, amount: float) -> dict:
    return {
        "name": f"Duman Test {suffix}",
        "identity_no": DEMO_TCKN[persona],
        "birth_date": "1985-01-01",
        "phone": f"0535{abs(hash(suffix)) % 10_000_000:07d}",
        "email": "smoke@example.com",
        "address": f"Smoke Sok. No:{suffix}",
        "iban": f"TR5500061000000000{abs(hash(suffix)) % 10**8:08d}",
        "monthly_income": income,
        "employment_type": "MAASLI",
        "employer_name": "Anadolu Bilişim",
        "product": "IHTIYAC",
        "requested_amount": amount,
        "requested_term_months": 36,
        "consents": {
            "kvkk_aydinlatma": True,
            "acik_riza": True,
            "kkb_sorgu": True,
            "edevlet_sorgu": True,
            "acik_bankacilik": True,
        },
    }


def wait_state(c: Client, app_id: str, targets: set[str], timeout: float) -> str:
    deadline = time.time() + timeout
    state = ""
    while time.time() < deadline:
        _, body = c.call(f"/api/v1/applications/{app_id}")
        state = body["state"]
        if state in targets:
            return state
        time.sleep(1.5)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    c = Client(args.base)
    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="anil2-smoke-"))
    run = uuid.uuid4().hex[:6]

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    s, body = c.call("/health/ready")
    check(
        "ready",
        s == 200,
        json.dumps(body.get("checks", {}), ensure_ascii=False) if isinstance(body, dict) else "",
    )

    def submit(persona: str, income: float, amount: float) -> str:
        c.login("basvuran")
        s, body = c.call(
            "/api/v1/applications", "POST", application(persona, f"{persona}{run}", income, amount)
        )
        assert s == 202, body
        app_id = body["application_id"]
        ob = open_banking_transactions(DEMO_TCKN[persona], income)
        files = generate_applicant_bundle(
            tmp / f"{persona}",
            name=f"Duman Test {persona}{run}",
            tckn=DEMO_TCKN[persona],
            iban="TR550006100000000000000001",
            address="Smoke Sok.",
            employer="Anadolu Bilişim",
            net_income=income,
            transactions=ob["transactions"],
            opening_balance=ob["account"]["opening_balance"],
        )
        for code, path in files.items():
            s, _ = c.upload(app_id, code, path)
            assert s == 201, (code, s)
        return app_id

    ok_id = submit("temiz", 45_000, 150_000)
    state = wait_state(
        c, ok_id, {"TEKLIF_SUNULDU", "UZMAN_INCELEMESI", "OTOMATIK_RET"}, args.timeout
    )
    check("clean applicant -> offer", state in ("TEKLIF_SUNULDU", "UZMAN_INCELEMESI"), state)
    if state == "TEKLIF_SUNULDU":
        s, body = c.call(f"/api/v1/applications/{ok_id}/offer/accept", "POST")
        check("offer accepted", s == 200 and body["state"] == "SOZLESME_HAZIR")
        c.login("uzman")
        s, body = c.call(f"/api/v1/workbench/{ok_id}/disburse", "POST")
        check("disbursed", s == 200 and body.get("state") == "KULLANDIRILDI")
        s, pdf = c.call(f"/api/v1/applications/{ok_id}/report?format=pdf")
        check("report pdf", s == 200 and isinstance(pdf, bytes) and pdf.startswith(b"%PDF"))
        s, body = c.call(f"/api/v1/agent/{ok_id}/memo", "POST")
        check(
            "AI memo",
            s == 201 and bool(body.get("sections")),
            body.get("mode", "") if isinstance(body, dict) else "",
        )

    grey_id = submit("gri", 42_000, 220_000)
    state = wait_state(
        c, grey_id, {"UZMAN_INCELEMESI", "TEKLIF_SUNULDU", "OTOMATIK_RET"}, args.timeout
    )
    check("grey zone -> review", state == "UZMAN_INCELEMESI", state)
    if state == "UZMAN_INCELEMESI":
        c.login("uzman")
        s, body = c.call(
            f"/api/v1/workbench/{grey_id}/decision",
            "POST",
            {
                "action": "ONAY",
                "amount": 320000,
                "justification": "Smoke: dört göz gerektiren onay önerisi.",
            },
        )
        check("maker submits", s == 201 and body["review"]["status"] == "ONAY_BEKLIYOR")
        review_id = body["review"]["review_id"]
        c.login("komite")
        s, body = c.call(
            f"/api/v1/workbench/reviews/{review_id}/check",
            "POST",
            {"approve": True, "note": "Smoke onayı."},
        )
        check("checker approves (four-eyes)", s == 200 and body["state"] == "TEKLIF_SUNULDU")

    bad_id = submit("gecikmeli", 40_000, 150_000)
    state = wait_state(c, bad_id, {"OTOMATIK_RET", "UZMAN_INCELEMESI"}, args.timeout)
    check("delinquent -> automatic rejection", state == "OTOMATIK_RET", state)
    if state == "OTOMATIK_RET":
        s, body = c.call(
            f"/api/v1/applications/{bad_id}/objection",
            "POST",
            {"reason": "Smoke: KVKK m.11 kapsamında insan incelemesi talebi."},
        )
        check("KVKK objection", s == 201 and body["state"] == "ITIRAZ_INCELEMESI")

    c.login("modelyon")
    for path in (
        "/api/v1/models",
        "/api/v1/governance/drift",
        "/api/v1/governance/fairness",
        "/api/v1/governance/champion-challenger",
        "/api/v1/llm/status",
    ):
        s, _ = c.call(path)
        check(path, s == 200)
    c.login("komite")
    s, body = c.call("/api/v1/audit/verify")
    check("audit chain valid", s == 200 and body.get("valid") is True)
    c.token = None
    s, _ = c.call("/")
    check("UI served", s == 200)
    s, _ = c.call("/metrics")
    check("prometheus metrics", s == 200)

    if failures:
        print(f"\nSMOKE FAILED: {len(failures)} -> {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nSMOKE PASS: all steps OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
