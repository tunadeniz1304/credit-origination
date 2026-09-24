"""Seed the six demo scenarios end to end (in-process, same code path as the API).

1. ``temiz``        — clean salaried applicant → automatic approval + offer
2. ``ince_dosya``   — thin bureau file, healthy cash flow → approved thanks to cash flow
3. ``yuksek_dsr``   — high debt service ratio → conditional counter-offer + counterfactual
4. ``kurcalanmis``  — tampered payslip → fraud signals + specialist review
5. ``halka``        — third applicant sharing phone/IBAN with two others → fraud-ring review
6. ``gri``          — grey-zone PD → specialist review, AI memorandum, pending four-eyes
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.security import Principal
from app.core.task_dispatcher import run_coroutine_safe
from app.db.models import Application, User
from app.db.session import session_scope
from app.documents.samples import generate_applicant_bundle
from app.integrations.personas import DEMO_TCKN, open_banking_transactions
from app.workflow.pipeline import Pipeline
from app.workflow.service import ApplicationService

CONSENTS = {
    "kvkk_aydinlatma": True,
    "acik_riza": True,
    "kkb_sorgu": True,
    "edevlet_sorgu": True,
    "acik_bankacilik": True,
}


@dataclass
class Scenario:
    key: str
    title: str
    persona: str
    name: str
    income: float
    amount: float
    term: int = 36
    tamper: bool = False
    phone: str = "05321234567"
    iban: str = "TR330006100519786457841326"
    expected: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    Scenario(
        "temiz",
        "Temiz maaşlı → otomatik onay",
        "temiz",
        "Elif Yıldırım",
        45_000,
        200_000,
        phone="05301110001",
        iban="TR120006400000011111111101",
        expected=("TEKLIF_SUNULDU",),
    ),
    Scenario(
        "ince_dosya",
        "İnce dosya + iyi nakit akışı → onay",
        "ince_dosya",
        "Kerem Aydın",
        35_000,
        150_000,
        phone="05301110002",
        iban="TR120006400000011111111102",
        expected=("TEKLIF_SUNULDU",),
    ),
    Scenario(
        "yuksek_dsr",
        "Yüksek DSR → koşullu karşı teklif",
        "yuksek_dsr",
        "Derya Şahin",
        45_000,
        650_000,
        phone="05301110003",
        iban="TR120006400000011111111103",
        expected=("TEKLIF_SUNULDU",),
    ),
    Scenario(
        "kurcalanmis",
        "Kurcalanmış bordro → sahtecilik incelemesi",
        "kurcalanmis",
        "Onur Çelik",
        45_000,
        250_000,
        tamper=True,
        phone="05301110004",
        iban="TR120006400000011111111104",
        expected=("UZMAN_INCELEMESI",),
    ),
    Scenario(
        "halka_2",
        "Halka üyesi (1/3)",
        "halka_2",
        "Mehmet Koç",
        40_000,
        120_000,
        phone="05309998877",
        iban="TR990006400000099999999999",
        expected=("TEKLIF_SUNULDU", "UZMAN_INCELEMESI"),
    ),
    Scenario(
        "halka_3",
        "Halka üyesi (2/3)",
        "halka_3",
        "Ahmet Er",
        40_000,
        130_000,
        phone="05309998877",
        iban="TR990006400000099999999999",
        expected=("TEKLIF_SUNULDU", "UZMAN_INCELEMESI"),
    ),
    Scenario(
        "halka",
        "Paylaşılan telefon/IBAN halkası → dolandırıcılık incelemesi",
        "halka",
        "Ali Tan",
        40_000,
        180_000,
        phone="05309998877",
        iban="TR990006400000099999999999",
        expected=("UZMAN_INCELEMESI",),
    ),
    Scenario(
        "gri",
        "Gri bölge → ajan memorandumu + dört göz",
        "gri",
        "Seda Arslan",
        42_000,
        220_000,
        phone="05301110006",
        iban="TR120006400000011111111106",
        expected=("UZMAN_INCELEMESI",),
        extra={"memo": True, "four_eyes": True},
    ),
]


def _address(scenario: Scenario) -> str:
    # Ring members share phone/IBAN only; every applicant has their own address.
    return f"Bağdat Cad. No:{10 + SCENARIOS.index(scenario)} Kadıköy İstanbul"


def _principal(session, username: str) -> Principal:
    user = session.execute(select(User).where(User.username == username)).scalar_one()
    return Principal(
        user_id=user.id, username=user.username, role=user.role, full_name=user.full_name
    )


def run_scenario(
    scenario: Scenario, settings: Settings | None = None, workdir: Path | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="anil2-demo-"))
    tckn = DEMO_TCKN[scenario.persona]
    ob = open_banking_transactions(tckn, scenario.income)
    files = generate_applicant_bundle(
        workdir / scenario.key,
        name=scenario.name,
        tckn=tckn,
        iban=scenario.iban,
        address=_address(scenario),
        employer="Anadolu Bilişim Ltd. Şti.",
        net_income=scenario.income,
        transactions=ob["transactions"],
        opening_balance=ob["account"]["opening_balance"],
        tamper_payslip=scenario.tamper,
    )
    with session_scope(settings) as session:
        owner = _principal(session, "basvuran")
        service = ApplicationService(session, owner, settings)
        app = service.create(
            {
                "name": scenario.name,
                "identity_no": tckn,
                "birth_date": None,
                "phone": scenario.phone,
                "email": f"{scenario.key}@example.com",
                "address": _address(scenario),
                "iban": scenario.iban,
                "gender": "K" if scenario.name.split()[0] in ("Elif", "Derya", "Seda") else "E",
                "province": "İstanbul",
                "monthly_income": scenario.income,
                "employment_type": "MAASLI",
                "employer_name": "Anadolu Bilişim Ltd. Şti.",
                "product": "IHTIYAC",
                "requested_amount": scenario.amount,
                "requested_term_months": scenario.term,
                "consents": CONSENTS,
                "persona": scenario.key,
            }
        )
        for code, path in files.items():
            service.add_document(app, code, path.name, path.read_bytes())
        application_id = app.id
    state = run_coroutine_safe(Pipeline(settings).run(application_id))
    result: dict[str, Any] = {
        "scenario": scenario.key,
        "title": scenario.title,
        "application_id": application_id,
        "state": state,
    }
    if scenario.extra.get("memo") or scenario.extra.get("four_eyes"):
        with session_scope(settings) as session:
            app = session.get(Application, application_id)  # type: ignore[assignment]
            if scenario.extra.get("memo"):
                from app.agents.underwriter.graph import UnderwriterAgent

                memo = run_coroutine_safe(UnderwriterAgent(session, app, actor="ajan:demo").run())
                app.letters = {**(app.letters or {}), "ai_memo": memo.model_dump()}
            if scenario.extra.get("four_eyes") and app.state == "UZMAN_INCELEMESI":
                from app.workbench.service import Workbench

                review = Workbench(session, _principal(session, "uzman")).submit_decision(
                    app,
                    action="ONAY",
                    justification="Gelir istikrarlı; kart borcunun kapatılması koşuluyla onay önerilir (demo).",
                )
                result["pending_review_id"] = review.id
                result["four_eyes"] = review.four_eyes
    result["as_expected"] = state in scenario.expected
    return result


def seed_demo(settings: Settings | None = None) -> list[dict[str, Any]]:
    workdir = Path(tempfile.mkdtemp(prefix="anil2-demo-"))
    return [run_scenario(s, settings, workdir) for s in SCENARIOS]
