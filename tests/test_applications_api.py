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


def test_schedule_endpoint_returns_plan_for_approved():
    response = client.post("/api/v1/applications", json=_payload())
    application_id = response.json()["application_id"]
    schedule = client.get(f"/api/v1/applications/{application_id}/schedule").json()
    assert schedule["term_months"] == 36
    assert schedule["principal"] == 1_800_000.0  # suggested_amount
    assert len(schedule["rows"]) == 36
    assert schedule["instalment"] > 0
    assert schedule["total_payment"] >= schedule["principal"]


def test_schedule_endpoint_rejects_unapproved():
    response = client.post(
        "/api/v1/applications",
        json=_payload(
            identity_no="34567890123", monthly_income=30_000,
            requested_amount=50_000, requested_term_months=24,
        ),
    )
    application_id = response.json()["application_id"]
    assert client.get(f"/api/v1/applications/{application_id}/schedule").status_code == 409


def test_upload_document_updates_completeness():
    submitted = client.post(
        "/api/v1/applications", json=_payload(submitted_documents=["IDENTITY"])
    )
    application_id = submitted.json()["application_id"]

    incomplete = client.post(
        f"/api/v1/applications/{application_id}/documents",
        files={"file": ("INCOME.txt", b"aylik gelir belgesi", "text/plain")},
        data={"code": "INCOME"},
    )
    assert incomplete.status_code == 200
    assert incomplete.json()["complete"] is False

    rest = ["EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"]
    for code in rest:
        client.post(
            f"/api/v1/applications/{application_id}/documents",
            files={"file": (f"{code}.txt", b"belge", "text/plain")},
            data={"code": code},
        )
    complete = client.get(f"/api/v1/applications/{application_id}").json()
    assert set(complete["application"]["applicant"]["submitted_documents"]) == set(
        ["IDENTITY", "INCOME", *rest]
    )


def test_upload_unknown_document_code_returns_400():
    submitted = client.post("/api/v1/applications", json=_payload())
    application_id = submitted.json()["application_id"]
    response = client.post(
        f"/api/v1/applications/{application_id}/documents",
        files={"file": ("X.txt", b"x", "text/plain")},
        data={"code": "NOT_A_CODE"},
    )
    assert response.status_code == 400


def test_reprocess_reenqueues_application():
    submitted = client.post("/api/v1/applications", json=_payload())
    application_id = submitted.json()["application_id"]
    response = client.post(f"/api/v1/applications/{application_id}/reprocess")
    assert response.status_code == 200
    body = response.json()
    assert body["application_id"] == application_id
    assert body["status"] in ("APPROVED", "REJECTED", "QUEUED", "PROCESSING")
