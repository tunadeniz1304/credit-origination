"""Workbench tests: authority matrix, four-eyes, overrides, objections, disbursal."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.security import Principal
from app.main import app
from app.workbench.authority import checker_allowed, evaluate_authority, required_role
from tests.helpers import login, submit_complete


def _p(role: str, name: str = "x") -> Principal:
    return Principal(user_id=name, username=name, role=role)


# ------------------------------------------------------------------ unit: authority matrix
@pytest.mark.parametrize(
    ("amount", "pd", "role"),
    [
        (100_000, 0.05, "uzman"),
        (300_000, 0.05, "kidemli_uzman"),
        (100_000, 0.15, "kidemli_uzman"),
        (900_000, 0.05, "komite"),
        (100_000, 0.5, "komite"),
    ],
)
def test_required_role(amount, pd, role):
    assert required_role(amount, pd) == role


def test_four_eyes_triggers():
    small = evaluate_authority(_p("uzman"), amount=100_000, pd=0.05, is_override=False)
    assert not small.four_eyes and small.maker_may_finalise
    big = evaluate_authority(_p("komite"), amount=400_000, pd=0.05, is_override=False)
    assert big.four_eyes and not big.maker_may_finalise
    override = evaluate_authority(_p("komite"), amount=50_000, pd=0.01, is_override=True)
    assert override.four_eyes and "override" in override.reasons[0]
    underpowered = evaluate_authority(_p("uzman"), amount=200_000, pd=0.11, is_override=False)
    assert underpowered.required_role == "kidemli_uzman" and not underpowered.maker_may_finalise


def test_checker_rules():
    assert checker_allowed(_p("komite", "a"), "a", 1) == (
        False,
        "dört göz ilkesi: talebi oluşturan kişi onaylayamaz",
    )
    assert not checker_allowed(_p("uzman", "b"), "a", 2)[0]
    assert checker_allowed(_p("kidemli_uzman", "b"), "a", 2) == (True, "")


# ------------------------------------------------------------------ API flows
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def users(client):
    return {
        name: login(client, name) for name in ("basvuran", "uzman", "uzman2", "kidemli", "komite")
    }


@pytest.fixture(scope="module")
def grey_id(client, users):
    application_id = submit_complete(
        client, users["basvuran"], "gri", monthly_income=42_000, requested_amount=220_000
    )
    assert (
        client.get(f"/api/v1/applications/{application_id}", headers=users["uzman"]).json()["state"]
        == "UZMAN_INCELEMESI"
    )
    return application_id


def test_queue_lists_grey_zone_with_sla(client, users, grey_id):
    queue = client.get("/api/v1/workbench/queue", headers=users["uzman"]).json()
    item = next(i for i in queue["items"] if i["application_id"] == grey_id)
    assert item["sla_remaining_hours"] > 20 and not item["sla_breached"]
    assert item["pd"] > 0.05 and item["reason_codes"]
    assert client.get("/api/v1/workbench/queue", headers=users["basvuran"]).status_code == 403
    assigned = client.post(f"/api/v1/workbench/{grey_id}/assign", headers=users["uzman"]).json()
    assert assigned["assigned_to"] == "uzman"
    mine = client.get("/api/v1/workbench/queue?mine=true", headers=users["uzman"]).json()
    assert grey_id in {i["application_id"] for i in mine["items"]}


def test_queue_pagination_search_and_filters(client, users, grey_id):
    second = submit_complete(
        client, users["basvuran"], "gri", monthly_income=40_000, requested_amount=210_000
    )
    full = client.get("/api/v1/workbench/queue?limit=100", headers=users["uzman"]).json()
    assert full["total"] >= 2 and full["offset"] == 0 and full["limit"] == 100
    first_page = client.get("/api/v1/workbench/queue?limit=1", headers=users["uzman"]).json()
    next_page = client.get(
        "/api/v1/workbench/queue?limit=1&offset=1", headers=users["uzman"]
    ).json()
    assert len(first_page["items"]) == 1 and len(next_page["items"]) == 1
    assert first_page["items"][0]["application_id"] != next_page["items"][0]["application_id"]
    assert first_page["total"] == full["total"]
    found = client.get(f"/api/v1/workbench/queue?q={second[-6:]}", headers=users["uzman"]).json()
    assert [i["application_id"] for i in found["items"]] == [second]
    assert found["items"][0]["product_label"] == "İhtiyaç Kredisi"
    none = client.get("/api/v1/workbench/queue?product=TASIT", headers=users["uzman"]).json()
    assert none["total"] == 0
    fresh = client.get("/api/v1/workbench/queue?sla_breached=false", headers=users["uzman"]).json()
    assert fresh["total"] == full["total"]
    assert client.get("/api/v1/workbench/queue?limit=500", headers=users["uzman"]).status_code == 422


def test_justification_is_mandatory(client, users, grey_id):
    response = client.post(
        f"/api/v1/workbench/{grey_id}/decision",
        json={"action": "ONAY", "justification": "ok"},
        headers=users["uzman"],
    )
    assert response.status_code == 422


def test_four_eyes_approval_flow(client, users, grey_id):
    preview = client.get(
        f"/api/v1/workbench/{grey_id}/authority?amount=320000", headers=users["uzman"]
    ).json()
    assert preview["four_eyes"] is True and preview["required_role"] == "kidemli_uzman"
    submitted = client.post(
        f"/api/v1/workbench/{grey_id}/decision",
        json={
            "action": "ONAY",
            "amount": 320_000,
            "justification": "Gelir istikrarlı, mevcut kart borcu kapatılacak; teminat yeterli.",
        },
        headers=users["uzman"],
    )
    assert submitted.status_code == 201, submitted.text
    review = submitted.json()["review"]
    assert review["status"] == "ONAY_BEKLIYOR" and review["four_eyes"]
    assert submitted.json()["state"] == "UZMAN_INCELEMESI"
    duplicate = client.post(
        f"/api/v1/workbench/{grey_id}/decision",
        json={"action": "RET", "justification": "İkinci karar denemesi yapılmamalı, test."},
        headers=users["uzman2"],
    )
    assert duplicate.status_code == 409
    same_person = client.post(
        f"/api/v1/workbench/reviews/{review['review_id']}/check",
        json={"approve": True, "note": "kendi onayım"},
        headers=users["uzman"],
    )
    assert same_person.status_code == 403
    low_rank = client.post(
        f"/api/v1/workbench/reviews/{review['review_id']}/check",
        json={"approve": True, "note": "yetkim yok"},
        headers=users["uzman2"],
    )
    assert low_rank.status_code == 403
    pending = client.get("/api/v1/workbench/reviews", headers=users["kidemli"]).json()["reviews"]
    assert review["review_id"] in {r["review_id"] for r in pending}
    approved = client.post(
        f"/api/v1/workbench/reviews/{review['review_id']}/check",
        json={"approve": True, "note": "Uygundur."},
        headers=users["kidemli"],
    )
    assert approved.status_code == 200 and approved.json()["state"] == "TEKLIF_SUNULDU"
    detail = client.get(f"/api/v1/applications/{grey_id}", headers=users["kidemli"]).json()
    assert (
        detail["decision"]["kind"] in ("manual", "override")
        and detail["decision"]["decided_by"] == "kidemli"
    )
    assert detail["offer"]["amount"] > 0


def test_override_of_automatic_rejection_requires_objection_then_human_review(client, users):
    application_id = submit_complete(
        client, users["basvuran"], "gecikmeli", monthly_income=40_000, requested_amount=120_000
    )
    assert (
        client.get(f"/api/v1/applications/{application_id}", headers=users["uzman"]).json()["state"]
        == "OTOMATIK_RET"
    )
    direct = client.post(
        f"/api/v1/workbench/{application_id}/decision",
        json={"action": "ONAY", "justification": "Doğrudan onay denemesi, izin verilmemeli."},
        headers=users["komite"],
    )
    assert direct.status_code == 409  # not in review
    client.post(
        f"/api/v1/applications/{application_id}/objection",
        json={"reason": "Gecikmeler hastalık döneminde oldu, belgelerimi sunabilirim."},
        headers=users["basvuran"],
    )
    queue = client.get(
        "/api/v1/workbench/queue?state=ITIRAZ_INCELEMESI", headers=users["uzman"]
    ).json()
    assert application_id in {i["application_id"] for i in queue["items"]}
    resolved = client.post(
        f"/api/v1/workbench/{application_id}/objection/resolve",
        json={"upheld": True, "note": "Sağlık raporu sunuldu, yeniden değerlendirilecek."},
        headers=users["uzman"],
    )
    assert resolved.json()["state"] == "UZMAN_INCELEMESI" and "kabul" in resolved.json()["letter"]
    override = client.post(
        f"/api/v1/workbench/{application_id}/decision",
        json={
            "action": "ONAY",
            "amount": 60_000,
            "justification": "Gecikmeler geçici sağlık sorunundan; tutar düşürülerek onay önerilir.",
        },
        headers=users["uzman"],
    )
    review = override.json()["review"]
    assert review["is_override"] and review["four_eyes"]
    rejected = client.post(
        f"/api/v1/workbench/reviews/{review['review_id']}/check",
        json={"approve": False, "note": "Risk kabul edilemez."},
        headers=users["komite"],
    )
    assert (
        rejected.json()["review"]["status"] == "REDDEDILDI"
        and rejected.json()["state"] == "UZMAN_INCELEMESI"
    )
    final = client.post(
        f"/api/v1/workbench/{application_id}/decision",
        json={
            "action": "RET",
            "justification": "Komite görüşü doğrultusunda başvuru reddedilmiştir.",
        },
        headers=users["uzman"],
    )
    final_review = final.json()["review"]
    assert final_review["four_eyes"]  # high PD: even a rejection needs a second pair of eyes
    closed = client.post(
        f"/api/v1/workbench/reviews/{final_review['review_id']}/check",
        json={"approve": True, "note": "Ret uygundur."},
        headers=users["komite"],
    )
    assert closed.json()["state"] == "REDDEDILDI"
    objections = client.get("/api/v1/workbench/objections", headers=users["uzman"]).json()[
        "objections"
    ]
    assert any(o["application_id"] == application_id and o["status"] == "KABUL" for o in objections)


def test_objection_rejected_keeps_decision(client, users):
    application_id = submit_complete(
        client, users["basvuran"], "gecikmeli", monthly_income=40_000, requested_amount=130_000
    )
    client.post(
        f"/api/v1/applications/{application_id}/objection",
        json={"reason": "Yeniden değerlendirme talep ediyorum lütfen."},
        headers=users["basvuran"],
    )
    resolved = client.post(
        f"/api/v1/workbench/{application_id}/objection/resolve",
        json={"upheld": False, "note": "Gecikme geçmişi güncel ve süreklilik gösteriyor."},
        headers=users["kidemli"],
    ).json()
    assert resolved["state"] == "REDDEDILDI" and "korunmuştur" in resolved["letter"]


def test_field_correction_and_disbursal(client, users):
    application_id = submit_complete(client, users["basvuran"], "temiz", requested_amount=120_000)
    docs = client.get(
        f"/api/v1/applications/{application_id}/documents", headers=users["uzman"]
    ).json()["documents"]
    field = next(f for d in docs for f in d["fields"] if f["name"] == "isveren")
    corrected = client.post(
        f"/api/v1/workbench/{application_id}/fields/{field['field_id']}",
        json={"value": "Anadolu Bilişim Ltd. Şti.", "note": "unvan doğrulandı"},
        headers=users["uzman"],
    ).json()
    assert corrected["confidence"] == 1.0 and corrected["corrected_by"] == "uzman"
    assert (
        client.post(
            f"/api/v1/workbench/{application_id}/disburse", headers=users["uzman"]
        ).status_code
        == 409
    )
    client.post(f"/api/v1/applications/{application_id}/offer/accept", headers=users["basvuran"])
    done = client.post(
        f"/api/v1/workbench/{application_id}/disburse", headers=users["uzman"]
    ).json()
    assert done["state"] == "KULLANDIRILDI"


def test_request_more_documents(client, users):
    application_id = submit_complete(
        client, users["basvuran"], "gri", monthly_income=42_000, requested_amount=210_000
    )
    response = client.post(
        f"/api/v1/workbench/{application_id}/decision",
        json={
            "action": "BELGE_ISTE",
            "justification": "Son 6 aya ait ek maaş bordrosu talep edilmektedir.",
        },
        headers=users["uzman"],
    ).json()
    assert response["state"] == "BELGE_BEKLENIYOR"
