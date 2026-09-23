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


def test_metrics_reports_status_distribution():
    body = client.get("/api/v1/metrics").json()
    assert "total_applications" in body
    assert "by_status" in body
    assert body["total_applications"] >= 1
    assert body["approved_count"] >= 1
    assert body["sum_suggested_amount"] > 0


def test_root_redirects_to_dashboard():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/static/dashboard.html"
    dashboard = client.get("/static/dashboard.html")
    assert dashboard.status_code == 200
    assert "Operasyon Paneli" in dashboard.text


def test_download_report_json_and_pdf():
    submitted = client.post("/api/v1/applications", json=_payload())
    application_id = submitted.json()["application_id"]
    j = client.get(f"/api/v1/applications/{application_id}/report")
    assert j.status_code == 200
    assert j.headers["content-type"].startswith("application/json")
    assert j.json()["status"] == "APPROVED"
    p = client.get(f"/api/v1/applications/{application_id}/report", params={"format": "pdf"})
    assert p.status_code == 200
    assert p.headers["content-type"] == "application/pdf"
    assert p.content[:4] == b"%PDF"
    assert client.get(
        f"/api/v1/applications/{application_id}/report", params={"format": "doc"}
    ).status_code == 400


def test_document_status_endpoint():
    complete = client.post("/api/v1/applications", json=_payload())
    body = client.get(f"/api/v1/applications/{complete.json()['application_id']}/documents").json()
    assert body["complete"] is True
    assert body["missing"] == []

    sparse = client.post("/api/v1/applications", json=_payload(submitted_documents=["IDENTITY"]))
    body = client.get(f"/api/v1/applications/{sparse.json()['application_id']}/documents").json()
    assert body["complete"] is False
    assert "INCOME" in body["missing"]
    assert body["request_draft"]


def test_scorecard_endpoint_returns_composite_grade():
    submitted = client.post("/api/v1/applications", json=_payload())
    application_id = submitted.json()["application_id"]
    body = client.get(f"/api/v1/applications/{application_id}/scorecard").json()
    assert body["application_id"] == application_id
    assert body["grade"] == "A"
    assert body["total_score"] == 100.0
    assert len(body["rows"]) == 4
    rejected = client.post(
        "/api/v1/applications",
        json=_payload(
            identity_no="34567890123", monthly_income=30_000,
            requested_amount=50_000, requested_term_months=24,
        ),
    ).json()
    rej_body = client.get(
        f"/api/v1/applications/{rejected['application_id']}/scorecard"
    ).json()
    assert rej_body["total_score"] < 100.0
    assert rej_body["grade"] != "A"


def test_offer_endpoint_returns_priced_terms():
    submitted = client.post("/api/v1/applications", json=_payload())
    application_id = submitted.json()["application_id"]
    body = client.get(f"/api/v1/applications/{application_id}/offer").json()
    assert body["status"] == "APPROVED"
    assert body["proposed_amount"] == 1_800_000.0
    assert body["instalment"] > 0
    assert body["risk_grade"] == "A"
    rejected = client.post(
        "/api/v1/applications",
        json=_payload(
            identity_no="34567890123", monthly_income=30_000,
            requested_amount=50_000, requested_term_months=24,
        ),
    ).json()
    assert client.get(
        f"/api/v1/applications/{rejected['application_id']}/offer"
    ).status_code == 409


def test_audit_endpoint_returns_lifecycle_events():
    submitted = client.post("/api/v1/applications", json=_payload())
    application_id = submitted.json()["application_id"]
    body = client.get(f"/api/v1/applications/{application_id}/audit").json()
    actions = [entry["action"] for entry in body["entries"]]
    assert "APPLICATION_SUBMITTED" in actions
    assert "APPLICATION_QUEUED" in actions
    # inline backend finalises synchronously, so the verdict is recorded.
    assert "APPLICATION_APPROVED" in actions
    assert len(body["entries"]) >= 3


def test_queue_status_reports_backend_and_registered_tasks():
    body = client.get("/api/v1/queue").json()
    assert body["backend"] == "inline"
    assert "app.tasks.process_application" in body["registered_tasks"]
    assert body["task_count"] >= 2


def test_document_chunks_retrieval():
    submitted = client.post(
        "/api/v1/applications", json=_payload(submitted_documents=["INCOME"])
    )
    application_id = submitted.json()["application_id"]
    content = ("Aylık gelir 30000 TRY, maaş ödemeleri banka hesap ekstresinden "
               "doğrulanır. Bu belgede bordro kesintileri ve primler listelenir.\n") * 20
    client.post(
        f"/api/v1/applications/{application_id}/documents",
        files={"file": ("INCOME.txt", content.encode(), "text/plain")},
        data={"code": "INCOME"},
    )
    body = client.get(
        f"/api/v1/applications/{application_id}/documents/INCOME/chunks",
        params={"query": "ekstresi", "k": 3},
    ).json()
    assert body["matched_file"]
    assert body["total_chunks"] >= 1
    assert body["chunks"]
    assert all("ekstresi" in chunk["text"].lower() for chunk in body["chunks"])


def test_notifications_outbox_reflects_approved_decision():
    client.post("/api/v1/applications", json=_payload())
    body = client.get("/api/v1/notifications").json()
    assert body["pending"] >= 1
    assert any(e["event"] == "APPROVED" for e in body["entries"])
    notif_id = body["entries"][0]["id"]
    delivered = client.post(f"/api/v1/notifications/{notif_id}/deliver").json()
    assert delivered["delivered"] is True
    after = client.get("/api/v1/notifications").json()
    assert sum(1 for e in after["entries"] if not e["delivered"]) < body["pending"]


def test_dispatch_task_drains_outbox():
    client.post("/api/v1/applications", json=_payload())
    pending_before = client.get("/api/v1/notifications").json()["pending"]
    resp = client.post("/api/v1/notifications/dispatch").json()
    assert resp["backend"] == "inline"
    dispatched = resp["result"]["dispatched"]
    assert dispatched >= 1
    pending_after = client.get("/api/v1/notifications").json()["pending"]
    assert pending_after == pending_before - dispatched
