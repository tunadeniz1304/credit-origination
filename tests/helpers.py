"""Shared helpers for API tests (auth, application payloads, document bundles)."""

from __future__ import annotations

import itertools
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi.testclient import TestClient

from app.documents.samples import generate_applicant_bundle
from app.integrations import personas
from app.integrations.personas import DEMO_TCKN, open_banking_transactions
from app.kyc.tckn import synthetic_tckn

DEMO_LOGIN = "Demo123!"  # public demo credential (dev only)
_TMP = Path(tempfile.mkdtemp(prefix="anil2-docs-"))


def login(client: TestClient, username: str) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login", json={"username": username, "password": DEMO_LOGIN}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


_counter = itertools.count(1)


def fresh_identity(persona: str) -> tuple[str, str, str]:
    """A new TCKN pinned to ``persona`` plus unique phone/IBAN (no velocity/ring hits)."""
    n = next(_counter)
    tckn = synthetic_tckn(f"test-{persona}-{n}-{id(_counter)}")
    personas._PINNED[tckn] = persona
    return tckn, f"0532{n:07d}", f"TR33000610051978{n:010d}"


def payload(persona: str = "temiz", **overrides) -> dict:
    tckn, phone, iban = fresh_identity(persona)
    body = {
        "name": "Ayşe Kaya",
        "identity_no": tckn,
        "birth_date": "1988-04-12",
        "phone": phone,
        "email": "ayse@example.com",
        "address": f"Moda Cad. No:{phone[-4:]} Kadıköy İstanbul",
        "iban": iban,
        "gender": "K",
        "province": "İstanbul",
        "monthly_income": 45_000,
        "employment_type": "MAASLI",
        "employer_name": "Anadolu Bilişim Ltd. Şti.",
        "product": "IHTIYAC",
        "requested_amount": 200_000,
        "requested_term_months": 36,
        "consents": {
            "kvkk_aydinlatma": True,
            "acik_riza": True,
            "kkb_sorgu": True,
            "edevlet_sorgu": True,
            "acik_bankacilik": True,
        },
    }
    body.update(overrides)
    return body


@lru_cache(maxsize=64)
def bundle(
    persona: str,
    income: float,
    tamper: bool = False,
    name: str = "Ayşe Kaya",
    iban: str = "TR330006100519786457841326",
    tckn: str | None = None,
) -> dict[str, Path]:
    tckn = tckn or DEMO_TCKN[persona]
    ob = open_banking_transactions(tckn, income)
    return generate_applicant_bundle(
        _TMP / f"{tckn}-{int(income)}-{tamper}-{abs(hash(name)) % 1000}",
        name=name,
        tckn=tckn,
        iban=iban,
        address="Moda Cad. No:5 Kadıköy İstanbul",
        employer="Anadolu Bilişim Ltd. Şti.",
        net_income=income,
        transactions=ob["transactions"],
        opening_balance=ob["account"]["opening_balance"],
        tamper_payslip=tamper,
    )


def upload_all(
    client: TestClient,
    headers: dict,
    application_id: str,
    files: dict[str, Path],
    skip: tuple[str, ...] = (),
) -> dict:
    last: dict = {}
    for code, path in files.items():
        if code in skip:
            continue
        with open(path, "rb") as fh:
            response = client.post(
                f"/api/v1/applications/{application_id}/documents",
                data={"code": code},
                files={"file": (path.name, fh, "application/pdf")},
                headers=headers,
            )
        assert response.status_code == 201, response.text
        last = response.json()
    return last


def submit_complete(
    client: TestClient, headers: dict, persona: str = "temiz", tamper: bool = False, **overrides
) -> str:
    body = payload(persona, **overrides)
    response = client.post("/api/v1/applications", json=body, headers=headers)
    assert response.status_code == 202, response.text
    application_id = response.json()["application_id"]
    upload_all(
        client,
        headers,
        application_id,
        bundle(
            persona, body["monthly_income"], tamper, body["name"], body["iban"], body["identity_no"]
        ),
    )
    return application_id
