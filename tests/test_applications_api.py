"""Application API tests: submit, fetch, validation (inline backend)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ALL_DOCS = ["IDENTITY", "INCOME", "EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"]


def _payload(**overrides) -> dict:
    base = {
        "name": "Ali Yılmaz",
        "identity_no": "12345678901",
        "monthly_income": 300_000,
        "requested_amount": 100_000,
        "requested_term_months": 36,
        "submitted_documents": ALL_DOCS,
    }
    base.update(overrides)
    return base


def test_health_reports_required_documents():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["queue_backend"] == "inline"
    assert "IDENTITY" in body["required_documents"]


def test_submit_approved_application_and_fetch():
    response = client.post("/api/v1/applications", json=_payload())
    assert response.status_code == 202
    body = response.json()
    application_id = body["application_id"]
    assert application_id.startswith("APP-")
    # inline backend executes synchronously: result is finalised immediately.
    assert body["status"] == "APPROVED"
    assert body["queue_backend"] == "inline"
    assert body["result"]["decision"]["suggested_amount"] == 1_800_000.0

    fetched = client.get(f"/api/v1/applications/{application_id}").json()
    assert fetched["application_id"] == application_id
    assert fetched["status"] == "APPROVED"
    assert fetched["result"]["status"] == "APPROVED"
    assert fitted_report_paths(fetched)


def fitted_report_paths(body: dict) -> bool:
    result = body["result"]
    return bool(result["report_json_path"] and result["report_pdf_path"])


def test_submit_rejected_application():
    response = client.post(
        "/api/v1/applications",
        json=_payload(
            identity_no="34567890123",
            monthly_income=30_000,
            requested_amount=50_000,
            requested_term_months=24,
        ),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "REJECTED"


def test_submit_invalid_payload_returns_422():
    response = client.post(
        "/api/v1/applications",
        json={"name": "x", "identity_no": "1", "monthly_income": 0,
              "requested_amount": 100, "requested_term_months": 12},
    )
    assert response.status_code == 422


def test_get_missing_application_returns_404():
    assert client.get("/api/v1/applications/APP-NOPE").status_code == 404


def test_list_applications_newest_first():
    body = client.get("/api/v1/applications").json()
    assert isinstance(body["applications"], list)
    ids = [item["application_id"] for item in body["applications"]]
    assert ids == sorted(ids, reverse=True)
