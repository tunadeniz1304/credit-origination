"""End-to-end browser flows against the real server (cookie session + CSRF + CSP).

* Applicant: log in, fill the wizard, upload the five documents, see the offer.
* Specialist + checker: find a grey-zone file via search, submit an approval
  that needs four eyes, log out, approve it as the senior specialist.
"""

from __future__ import annotations

import time

import httpx
import pytest

from app.integrations.personas import DEMO_TCKN
from tests.helpers import DEMO_LOGIN, bundle

pytestmark = pytest.mark.e2e

NAME = "Ayşe Kaya"
ADDRESS = "Moda Cad. No:5 Kadıköy İstanbul"


def _api_login(base: str, username: str) -> dict[str, str]:
    response = httpx.post(
        base + "/api/v1/auth/login", json={"username": username, "password": DEMO_LOGIN}
    )
    response.raise_for_status()
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def _wait_for_state(base: str, headers: dict, application_id: str, states: set[str]) -> str:
    deadline = time.monotonic() + 90
    state = ""
    while time.monotonic() < deadline:
        state = httpx.get(f"{base}/api/v1/applications/{application_id}", headers=headers).json()[
            "state"
        ]
        if state in states:
            return state
        time.sleep(0.5)
    raise AssertionError(f"{application_id} stuck in {state}, expected {states}")


def test_applicant_applies_uploads_documents_and_sees_offer(ui) -> None:
    page = ui.page
    iban = "TR330006100519786457841326"
    ui.login("basvuran")
    ui.assert_session_cookie_is_http_only()
    assert page.evaluate("() => localStorage.length + sessionStorage.length") == 0

    page.get_by_role("button", name="Yeni başvuru").click()
    page.fill("#f-name", NAME)
    page.fill("#f-tckn", DEMO_TCKN["temiz"])
    page.fill("#f-phone", "05329876543")
    page.fill("#f-iban", iban)
    page.fill("#f-address", ADDRESS)
    page.get_by_role("button", name="İleri").click()
    page.fill("#f-income", "45000")
    page.select_option("#f-employment", "MAASLI")
    page.select_option("#f-product", "IHTIYAC")
    page.fill("#f-amount", "200000")
    page.fill("#f-term", "36")
    assert page.locator("#f-product option:checked").inner_text() == "İhtiyaç Kredisi"
    page.get_by_role("button", name="İleri").click()
    for consent in ("#c-kvkk", "#c-riza", "#c-kkb", "#c-edevlet", "#c-ob"):
        page.check(consent)
    page.get_by_role("button", name="Başvuruyu gönder").click()

    page.get_by_role("heading", name="Belgelerim").wait_for()
    files = bundle("temiz", 45_000, name=NAME, iban=iban, tckn=DEMO_TCKN["temiz"])
    assert len(files) == 5
    for code, path in files.items():
        page.select_option("#doc-code", code)
        with page.expect_response(
            lambda r: r.url.endswith("/documents") and r.request.method == "POST"
        ) as upload:
            page.set_input_files("#doc-file", str(path))
        assert upload.value.status == 201, upload.value.text()

    amount = page.get_by_test_id("offer-amount")
    amount.wait_for(timeout=90_000)
    assert "₺" in amount.inner_text() and "200.000" in amount.inner_text()
    assert "₺" in page.get_by_test_id("offer-instalment").inner_text()
    assert page.get_by_role("button", name="Teklifi kabul et").is_visible()
    body = page.locator("main").inner_text()
    for raw in ("IHTIYAC", "MAASLI", "TEKLIF_SUNULDU"):
        assert raw not in body


def test_specialist_override_then_senior_approves_four_eyes(ui) -> None:
    base, page = ui.base, ui.page
    applicant = _api_login(base, "basvuran")
    iban = "TR330006100519786400000042"
    body = {
        "name": NAME,
        "identity_no": DEMO_TCKN["gri"],
        "birth_date": "1988-04-12",
        "phone": "05321110042",
        "email": "ayse@example.com",
        "address": ADDRESS,
        "iban": iban,
        "gender": "K",
        "province": "İstanbul",
        "monthly_income": 42_000,
        "employment_type": "MAASLI",
        "employer_name": "Anadolu Bilişim Ltd. Şti.",
        "product": "IHTIYAC",
        "requested_amount": 220_000,
        "requested_term_months": 36,
        "consents": dict.fromkeys(
            ("kvkk_aydinlatma", "acik_riza", "kkb_sorgu", "edevlet_sorgu", "acik_bankacilik"),
            True,
        ),
    }
    created = httpx.post(base + "/api/v1/applications", json=body, headers=applicant)
    assert created.status_code == 202, created.text
    application_id = created.json()["application_id"]
    for code, path in bundle("gri", 42_000, name=NAME, iban=iban, tckn=DEMO_TCKN["gri"]).items():
        with open(path, "rb") as fh:
            response = httpx.post(
                f"{base}/api/v1/applications/{application_id}/documents",
                data={"code": code},
                files={"file": (path.name, fh, "application/pdf")},
                headers=applicant,
            )
        assert response.status_code == 201, response.text
    _wait_for_state(base, applicant, application_id, {"UZMAN_INCELEMESI"})

    # Maker: the specialist finds the file through the debounced search box.
    ui.login("uzman")
    ui.assert_session_cookie_is_http_only()
    page.fill("#q-search", application_id[-6:])
    row = page.locator("button.queue-open", has_text=application_id)
    row.wait_for()
    page.wait_for_function(
        "() => document.querySelectorAll('button.queue-open').length === 1", polling=200
    )
    assert "1–1 / 1" in page.locator(".pager").inner_text()
    row.click()
    page.locator("#staff-title", has_text=application_id).wait_for()
    page.get_by_role("tab", name="Karar ver").click()
    page.select_option("#dec-action", "ONAY")
    page.fill("#dec-amount", "320000")
    page.fill(
        "#dec-just",
        "Gelir istikrarlı, mevcut kart borcu kapatılacak; teminat ve ödeme geçmişi yeterli.",
    )
    page.get_by_role("button", name="Kararı gönder").click()
    page.locator(".toast", has_text="Dört göz onayı bekleniyor").wait_for()
    ui.logout()

    # Checker: the authority matrix requires a senior specialist for 320k at this PD.
    ui.login("kidemli")
    approve = page.get_by_role("button", name=f"Onayla: {application_id}")
    approve.wait_for()
    approve.click()
    page.locator(".toast", has_text="Teklif sunuldu").wait_for()
    staff = _api_login(base, "kidemli")
    assert _wait_for_state(base, staff, application_id, {"TEKLIF_SUNULDU"}) == "TEKLIF_SUNULDU"
    detail = httpx.get(f"{base}/api/v1/applications/{application_id}", headers=staff).json()
    assert detail["offer"]["amount"] > 0 and detail["decision"]["decided_by"] == "kidemli"
