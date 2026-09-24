"""The six demo scenarios deliver their expected outcomes (integration test)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.helpers import login

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


@pytest.fixture(scope="module")
def seeded():
    with TestClient(app) as client:
        admin = login(client, "admin")
        response = client.post("/api/v1/demo/seed", headers=admin)
        assert response.status_code == 200, response.text
        yield client, {r["scenario"]: r for r in response.json()["results"]}


def test_all_scenarios_as_expected(seeded):
    _, results = seeded
    assert all(r["as_expected"] for r in results.values()), results


def test_clean_and_thin_file_get_offers(seeded):
    client, results = seeded
    uzman = login(client, "uzman")
    for key in ("temiz", "ince_dosya"):
        detail = client.get(
            f"/api/v1/applications/{results[key]['application_id']}", headers=uzman
        ).json()
        assert (
            detail["state"] == "TEKLIF_SUNULDU" and detail["decision"]["outcome"] == "OTOMATIK_ONAY"
        )
    thin = client.get(
        f"/api/v1/applications/{results['ince_dosya']['application_id']}/decision", headers=uzman
    ).json()
    assert (
        thin["feature_snapshot"]["bureau_hit"] == 0
        and thin["feature_snapshot"]["cashflow_available"] == 1
    )


def test_high_dsr_is_conditional_with_counterfactual(seeded):
    client, results = seeded
    decision = client.get(
        f"/api/v1/applications/{results['yuksek_dsr']['application_id']}/decision",
        headers=login(client, "uzman"),
    ).json()
    assert decision["conditional"] is True
    assert decision["limits"]["offer_amount"] < 650_000
    assert decision["counterfactuals"]


def test_tampered_payslip_flagged(seeded):
    client, results = seeded
    uzman = login(client, "uzman")
    application_id = results["kurcalanmis"]["application_id"]
    decision = client.get(f"/api/v1/applications/{application_id}/decision", headers=uzman).json()
    assert "R11_SAHTECILIK_SUPHESI" in [r["code"] for r in decision["reason_codes"]]
    docs = client.get(f"/api/v1/applications/{application_id}/documents", headers=uzman).json()[
        "documents"
    ]
    income = next(d for d in docs if d["code"] == "INCOME")
    assert income["status"] == "SUPHELI" and income["fraud_score"] >= 0.6


def test_fraud_ring_flagged_with_graph(seeded):
    client, results = seeded
    uzman = login(client, "uzman")
    application_id = results["halka"]["application_id"]
    decision = client.get(f"/api/v1/applications/{application_id}/decision", headers=uzman).json()
    assert "R15_HALKA_SUPHESI" in [r["code"] for r in decision["reason_codes"]]
    network = client.get(f"/api/v1/applications/{application_id}/network", headers=uzman).json()
    assert (
        network["flagged"]
        and network["ring_size"] >= 3
        and set(network["shared_identifiers"]) >= {"phone", "iban"}
    )


def test_grey_zone_has_memo_and_pending_four_eyes(seeded):
    client, results = seeded
    uzman = login(client, "uzman")
    grey = results["gri"]
    assert grey["four_eyes"] is True and grey["pending_review_id"]
    memo = client.get(f"/api/v1/agent/{grey['application_id']}/memo", headers=uzman).json()
    assert memo["sections"] and memo["citations"]
    pending = client.get("/api/v1/workbench/reviews", headers=uzman).json()["reviews"]
    assert grey["pending_review_id"] in {r["review_id"] for r in pending}


def test_frontend_never_uses_innerhtml():
    """Stored-XSS regression (bug #2): user data is only rendered via textContent/x-text."""
    source = (STATIC / "app.js").read_text(encoding="utf-8") + (STATIC / "index.html").read_text(
        encoding="utf-8"
    )
    assert not re.search(r"\.innerHTML\s*=|x-html|insertAdjacentHTML|document\.write", source)
    assert not (STATIC / "dashboard.html").exists()


def test_index_served(seeded):
    client, _ = seeded
    page = client.get("/")
    assert page.status_code == 200 and "Kredi Tahsis Platformu" in page.text
    assert client.get("/static/vendor/alpine.min.js").status_code == 200
