"""End-to-end API tests (inline backend, SQLite, demo LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.reports.credit_report import pdf_text
from tests.helpers import bundle, login, payload, submit_complete, upload_all


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def applicant(client):
    return login(client, "basvuran")


@pytest.fixture(scope="module")
def other_applicant(client):
    return login(client, "basvuran2")


@pytest.fixture(scope="module")
def specialist(client):
    return login(client, "uzman")


@pytest.fixture(scope="module")
def approved_id(client, applicant):
    return submit_complete(client, applicant, "temiz")


# ------------------------------------------------------------------ system
def test_health_and_probes(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["queue_backend"] == "inline"
    assert "IDENTITY" in body["required_documents"]
    assert client.get("/health/live").json() == {"status": "live"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] == "ok"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and "anil2_applications_submitted_total" in metrics.text


def test_security_headers_and_request_id(client):
    response = client.get("/health/live", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Frame-Options"] == "DENY"


# ------------------------------------------------------------------ auth / RBAC
def test_endpoints_require_authentication(client):
    # Earlier logins in this module left an HttpOnly session cookie on the shared client.
    client.cookies.clear()
    assert client.get("/api/v1/applications").status_code == 401
    assert client.post("/api/v1/applications", json=payload()).status_code == 401
    assert client.get("/api/v1/metrics").status_code == 401
    assert client.post("/api/v1/notifications/dispatch").status_code == 401


def test_login_failure_and_me(client, applicant):
    bad = client.post("/api/v1/auth/login", json={"username": "uzman", "password": "yanlis-parola"})
    assert bad.status_code == 401
    me = client.get("/api/v1/auth/me", headers=applicant).json()
    assert me["role"] == "basvuran" and me["is_staff"] is False


def test_register_applicant(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "yeni_kisi", "password": "Parola123!", "full_name": "Yeni Kişi"},
    )
    assert response.status_code == 201 and response.json()["role"] == "basvuran"
    assert (
        client.post(
            "/api/v1/auth/register",
            json={"username": "yeni_kisi", "password": "Parola123!", "full_name": "Yeni Kişi"},
        ).status_code
        == 409
    )


def test_role_restrictions(client, applicant):
    assert client.get("/api/v1/metrics", headers=applicant).status_code == 403
    assert client.post("/api/v1/notifications/dispatch", headers=applicant).status_code == 403
    modelyon = login(client, "modelyon")
    assert client.post("/api/v1/applications", json=payload(), headers=modelyon).status_code == 403


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"identity_no": "12345678901"}, "checksum"),
        ({"identity_no": "123"}, "identity_no"),
        ({"iban": "TR12"}, "IBAN"),
        ({"phone": "12345"}, "telefon"),
        ({"monthly_income": -1}, "monthly_income"),
        ({"currency": "USD"}, "currency"),
        ({"consents": {"kvkk_aydinlatma": True, "acik_riza": False, "kkb_sorgu": True}}, "rıza"),
    ],
)
def test_submission_validation(client, applicant, override, fragment):
    response = client.post("/api/v1/applications", json=payload(**override), headers=applicant)
    assert response.status_code == 422
    assert fragment.lower() in response.text.lower()


# ------------------------------------------------------------------ document gate
def test_missing_documents_stop_the_flow_with_a_letter(client, applicant):
    response = client.post("/api/v1/applications", json=payload("temiz"), headers=applicant)
    assert response.status_code == 202
    application_id = response.json()["application_id"]
    detail = client.get(f"/api/v1/applications/{application_id}", headers=applicant).json()
    assert detail["state"] == "BELGE_BEKLENIYOR"  # bug #3 regression
    assert set(detail["missing_documents"]) == {
        "IDENTITY",
        "INCOME",
        "EMPLOYMENT",
        "ADDRESS",
        "BANK_STATEMENT",
    }
    letter = detail["letters"]["eksik_belge"]
    assert "Maaş Bordrosu" in letter and "[mock-llm]" not in letter
    states = [e["to"] for e in detail["timeline"]]
    assert states == ["TASLAK", "GONDERILDI", "BELGE_BEKLENIYOR"]

    files = bundle("temiz", 45_000)
    partial = upload_all(client, applicant, application_id, files, skip=("BANK_STATEMENT",))
    assert partial["missing_documents"] == ["BANK_STATEMENT"] and not partial["processing_resumed"]
    final = upload_all(
        client, applicant, application_id, {"BANK_STATEMENT": files["BANK_STATEMENT"]}
    )
    assert final["complete"] and final["processing_resumed"]
    assert (
        client.get(f"/api/v1/applications/{application_id}", headers=applicant).json()["state"]
        == "TEKLIF_SUNULDU"
    )


# ------------------------------------------------------------------ happy path
def test_clean_applicant_gets_offer(client, applicant, approved_id):
    detail = client.get(f"/api/v1/applications/{approved_id}", headers=applicant).json()
    assert detail["state"] == "TEKLIF_SUNULDU"
    assert detail["applicant"]["identity_no_masked"].startswith(
        detail["applicant"]["identity_no_masked"][:2]
    )
    assert "*" in detail["applicant"]["identity_no_masked"]
    decision = detail["decision"]
    assert decision["outcome"] == "OTOMATIK_ONAY"
    assert "pd" not in decision  # applicants never see model internals
    offer = detail["offer"]
    assert offer["amount"] == 200_000 and offer["instalment"] > 0 and len(offer["schedule"]) == 36
    timeline = [e["to"] for e in detail["timeline"]]
    assert timeline[-5:] == [
        "BELGE_INCELEMEDE",
        "VERI_TOPLANIYOR",
        "KARAR_MOTORU",
        "OTOMATIK_ONAY",
        "TEKLIF_SUNULDU",
    ]


def test_staff_sees_full_decision_and_cashflow(client, specialist, approved_id):
    decision = client.get(f"/api/v1/applications/{approved_id}/decision", headers=specialist).json()
    assert 0 < decision["pd"] < 0.05
    assert decision["rule_set_version"] == "policy_v2" and decision["model_version"] == "pd_lgbm_v2"
    assert decision["shap"] and decision["scorecard"]["points"] >= 300
    assert decision["narratives"]["mode"] == "demo"
    assert "KREDİ KOMİTESİ ÖZETİ" in decision["narratives"]["committee_summary"]
    cashflow = client.get(f"/api/v1/applications/{approved_id}/cashflow", headers=specialist).json()
    assert cashflow["available"] and len(cashflow["monthly"]) == 12
    docs = client.get(f"/api/v1/applications/{approved_id}/documents", headers=specialist).json()
    income = next(d for d in docs["documents"] if d["code"] == "INCOME")
    assert income["fraud_score"] == 0.0 and any(f["name"] == "net_ucret" for f in income["fields"])


def test_report_pdf_has_turkish_glyphs(client, specialist, approved_id):
    response = client.get(
        f"/api/v1/applications/{approved_id}/report?format=pdf", headers=specialist
    )
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    path = Path(client.app_state_tmp) if hasattr(client, "app_state_tmp") else None
    tmp = Path(__file__).parent / "_report_tmp.pdf"
    tmp.write_bytes(response.content)
    try:
        text = pdf_text(tmp)
    finally:
        tmp.unlink()
    assert "KREDİ TAHSİS MEMORANDUMU" in text  # bug #11 regression: İ survives
    for glyph in "ığşİ":
        assert glyph in text
    # F04: user-facing labels, never raw enum codes.
    assert "İhtiyaç Kredisi" in text and "Maaşlı çalışan" in text
    for raw in ("IHTIYAC", "MAASLI", "Karar türü: engine"):
        assert raw not in text, raw
    json_report = client.get(
        f"/api/v1/applications/{approved_id}/report?format=json", headers=specialist
    ).json()
    assert json_report["decision"]["outcome"] == "OTOMATIK_ONAY"
    assert path is None


def test_offer_acceptance_produces_contract(client, applicant):
    application_id = submit_complete(client, applicant, "temiz", requested_amount=150_000)
    accepted = client.post(f"/api/v1/applications/{application_id}/offer/accept", headers=applicant)
    assert accepted.status_code == 200 and accepted.json()["state"] == "SOZLESME_HAZIR"
    contract = client.get(f"/api/v1/applications/{application_id}/contract", headers=applicant)
    assert contract.status_code == 200 and contract.content.startswith(b"%PDF")
    again = client.post(f"/api/v1/applications/{application_id}/offer/accept", headers=applicant)
    assert again.status_code == 409


def test_replay_is_deterministic(client, specialist, approved_id):
    decision = client.get(f"/api/v1/applications/{approved_id}/decision", headers=specialist).json()
    replay = client.post(
        f"/api/v1/decisions/{decision['decision_id']}/replay", headers=specialist
    ).json()
    assert replay["identical"] is True
    assert replay["original"] == replay["replayed"]


# ------------------------------------------------------------------ adverse path + KVKK m.11
def test_rejection_letter_and_objection(client, applicant, specialist):
    application_id = submit_complete(
        client, applicant, "gecikmeli", monthly_income=40_000, requested_amount=150_000
    )
    detail = client.get(f"/api/v1/applications/{application_id}", headers=applicant).json()
    assert detail["state"] == "OTOMATIK_RET"
    decision = detail["decision"]
    assert decision["objection_right"] is True and decision["reason_codes"]
    letter = decision["applicant_letter"]
    assert "itiraz" in letter.lower() and "6698 sayılı" in letter
    assert (
        client.get(f"/api/v1/applications/{application_id}/offer", headers=applicant).status_code
        == 404
    )
    short = client.post(
        f"/api/v1/applications/{application_id}/objection",
        json={"reason": "kısa"},
        headers=applicant,
    )
    assert short.status_code == 422
    objection = client.post(
        f"/api/v1/applications/{application_id}/objection",
        json={"reason": "Gelirim son aylarda arttı, yeniden değerlendirilmesini talep ediyorum."},
        headers=applicant,
    )
    assert objection.status_code == 201 and objection.json()["state"] == "ITIRAZ_INCELEMESI"


def test_objection_only_for_adverse(client, applicant, approved_id):
    response = client.post(
        f"/api/v1/applications/{approved_id}/objection",
        json={"reason": "Olumlu karara itiraz edilemez, test."},
        headers=applicant,
    )
    assert response.status_code == 409


# ------------------------------------------------------------------ isolation / ownership
def test_applicant_cannot_see_others(client, other_applicant, approved_id):
    assert (
        client.get(f"/api/v1/applications/{approved_id}", headers=other_applicant).status_code
        == 404
    )
    listing = client.get("/api/v1/applications", headers=other_applicant).json()
    assert approved_id not in {a["application_id"] for a in listing["applications"]}


def test_uploads_are_isolated_per_application(client, applicant, specialist):
    first = client.post(
        "/api/v1/applications", json=payload("temiz", name="Birinci Kişi"), headers=applicant
    ).json()["application_id"]
    second = client.post(
        "/api/v1/applications", json=payload("temiz", name="İkinci Kişi"), headers=applicant
    ).json()["application_id"]
    a = upload_all(
        client, applicant, first, {"INCOME": bundle("temiz", 45_000, name="Birinci Kişi")["INCOME"]}
    )
    b = upload_all(
        client, applicant, second, {"INCOME": bundle("temiz", 60_000, name="İkinci Kişi")["INCOME"]}
    )
    assert a["document"]["sha256"] != b["document"]["sha256"]
    chunks = client.get(
        f"/api/v1/applications/{first}/documents/{a['document']['document_id']}/chunks?query=Ad Soyad",
        headers=specialist,
    ).json()
    text = " ".join(c["text"] for c in chunks["chunks"])
    assert "Birinci" in text and "İkinci" not in text  # bug #1 regression (no leakage)
    cross = client.get(
        f"/api/v1/applications/{second}/documents/{a['document']['document_id']}/file",
        headers=specialist,
    )
    assert cross.status_code == 404


def test_upload_rejects_executables(client, applicant):
    application_id = client.post(
        "/api/v1/applications", json=payload("temiz"), headers=applicant
    ).json()["application_id"]
    response = client.post(
        f"/api/v1/applications/{application_id}/documents",
        data={"code": "INCOME"},
        files={"file": ("evil.pdf", b"MZ\x90\x00\x03\x00binary\x00\x00", "application/pdf")},
        headers=applicant,
    )
    assert response.status_code == 422
    unknown = client.post(
        f"/api/v1/applications/{application_id}/documents",
        data={"code": "PASAPORT_XYZ"},
        files={"file": ("a.pdf", b"%PDF-1.4 test", "application/pdf")},
        headers=applicant,
    )
    assert unknown.status_code == 400


def test_listing_is_newest_first(client, specialist):
    rows = client.get("/api/v1/applications?limit=200", headers=specialist).json()["applications"]
    created = [r["created_at"] for r in rows]
    assert created == sorted(created, reverse=True)  # bug #10 regression


# ------------------------------------------------------------------ observability / governance
def test_db_metrics_llm_status_and_audit_chain(client, specialist, approved_id):
    # approved_id guarantees one auto-approved decision whatever the test order.
    metrics = client.get("/api/v1/metrics", headers=specialist).json()
    assert metrics["total_applications"] >= 1
    assert metrics["decisions"].get("OTOMATIK_ONAY", 0) >= 1
    status = client.get("/api/v1/llm/status", headers=specialist).json()
    assert status["mode"] == "demo" and status["key_present"] is False and status["calls"] >= 1
    komite = login(client, "komite")
    chain = client.get("/api/v1/audit/verify", headers=komite).json()
    assert chain["valid"] is True and chain["entries"] > 10


def test_audit_trail_has_actors(client, specialist, approved_id):
    entries = client.get(f"/api/v1/applications/{approved_id}/audit", headers=specialist).json()[
        "entries"
    ]
    actions = [e["action"] for e in entries]
    assert "APPLICATION_CREATED" in actions and "DECISION_MADE" in actions
    assert {e["actor"] for e in entries} >= {"basvuran", "system"}


def test_notifications_outbox(client, specialist):
    """Self-contained: other tests' inline background work may add messages concurrently."""
    import uuid

    from app.db.models import OutboxMessage
    from app.db.outbox import enqueue
    from app.db.session import session_scope

    tag = uuid.uuid4().hex[:8]
    with session_scope() as session:
        ids = [
            enqueue(
                session, event="TEST", aggregate_id=f"{tag}-{i}", payload={}, channel="console"
            ).id
            for i in range(2)
        ]
    admin = login(client, "admin")
    listing = client.get("/api/v1/notifications", headers=specialist).json()
    assert listing["pending"] >= 2
    deliver = f"/api/v1/notifications/{ids[0]}/deliver"
    assert client.post(deliver, headers=admin).json()["delivered"] is True
    assert client.post(deliver, headers=admin).status_code == 404
    assert client.post("/api/v1/notifications/dispatch", headers=admin).json()["sent"] >= 1
    client.post("/api/v1/notifications/dispatch", headers=admin)  # idempotent for sent messages
    with session_scope(readonly=True) as session:
        second = session.get(OutboxMessage, ids[1])
        assert second.status == "SENT" and second.attempts == 1


def test_pricing_quote(client, applicant, specialist):
    quote = client.post(
        "/api/v1/pricing/quote",
        json={"amount": 100_000, "term_months": 24, "risk_band": "A"},
        headers=applicant,
    ).json()
    assert quote["annual_rate"] > 0 and quote["pd_source"] == "bant A"
    staff = client.post(
        "/api/v1/pricing/quote",
        json={"amount": 100_000, "term_months": 24, "pd": 0.1},
        headers=specialist,
    ).json()
    assert staff["pd"] == 0.1 and staff["annual_rate"] > quote["annual_rate"]
    bad = client.post(
        "/api/v1/pricing/quote", json={"amount": 100_000, "term_months": 100}, headers=applicant
    )
    assert bad.status_code == 422


def test_queue_and_cancel(client, applicant, specialist):
    queue = client.get("/api/v1/queue", headers=specialist).json()
    assert "app.tasks.process_application" in queue["registered_tasks"]
    application_id = client.post(
        "/api/v1/applications", json=payload("temiz"), headers=applicant
    ).json()["application_id"]
    cancelled = client.post(
        f"/api/v1/applications/{application_id}/cancel", headers=applicant
    ).json()
    assert cancelled["state"] == "IPTAL"
    assert (
        client.post(f"/api/v1/applications/{application_id}/cancel", headers=applicant).status_code
        == 409
    )
