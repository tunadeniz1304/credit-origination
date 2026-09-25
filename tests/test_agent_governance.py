"""Agent (fake LLM tool use, ReAct fallback, citations), RAG and governance tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agents.llm import DeterministicLLMProvider, LLMResponse, ToolCall, ToolsNotSupportedError
from app.agents.llm_service import LLMService
from app.agents.policy_assistant import ask, extractive_answer
from app.agents.rag import PolicyKnowledgeBase, tokenize
from app.agents.underwriter.graph import MAX_STEPS, UnderwriterAgent, _parse_react
from app.core.config import Settings
from app.db.models import Application
from app.db.session import session_scope
from app.governance.drift import compute_drift, psi
from app.main import app
from tests.helpers import login, submit_complete

LIVE = Settings(
    _env_file=None, llm_mode="live", llm_api_key="fake-key", llm_base_url="https://llm.test/v1"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def users(client):
    return {n: login(client, n) for n in ("basvuran", "uzman", "modelyon", "modelyon2", "komite")}


@pytest.fixture(scope="module")
def app_id(client, users):
    return submit_complete(client, users["basvuran"], "temiz", requested_amount=180_000)


# ------------------------------------------------------------------ RAG
def test_policy_kb_retrieves_dsr_section():
    kb = PolicyKnowledgeBase.load()
    assert kb.size > 10
    hits = kb.search("borç servis oranı sınırı yüzde kaç", k=3)
    assert (hits and "Borç servis oranı" in hits[0]["text"]) or "DSR" in hits[0]["title"]


def test_tokenize_folds_turkish():
    assert tokenize("Borç Servis ORANI") == ["borc", "servis", "orani"]


async def test_policy_assistant_demo_has_citations():
    result = await ask(
        "KVKK m.11 itiraz hakkı nedir?", LLMService(Settings(_env_file=None, llm_mode="demo"))
    )
    assert result["mode"] == "demo" and result["cited"]
    assert set(result["cited"]) <= {s["chunk_id"] for s in result["sources"]}


async def test_policy_assistant_rejects_invented_sources():
    provider = DeterministicLLMProvider(["Yanıt [kaynak:uydurma#1]", "Yine [kaynak:uydurma#2]"])
    result = await ask("dört göz ilkesi", LLMService(LIVE, provider=provider, mode="live"))
    assert result["mode"] == "fallback" and all(c != "uydurma#1" for c in result["cited"])


def test_extractive_answer_without_hits():
    assert "bulunamadı" in extractive_answer("x", [])


# ------------------------------------------------------------------ agent
def test_react_parser():
    assert _parse_react('Eylem: get_pricing\nGirdi: {"a": 1}') == ("get_pricing", {"a": 1})
    assert _parse_react("Action: search_policy") == ("search_policy", {})
    assert _parse_react("sadece metin") is None


async def test_demo_agent_visits_tools_and_cites_every_number(app_id):
    with session_scope() as session:
        application = session.get(Application, app_id)
        memo = await UnderwriterAgent(
            session, application, LLMService(Settings(_env_file=None, llm_mode="demo"))
        ).run()
        facts = {
            fid: f.value for fid, f in UnderwriterAgent(session, application).tools.facts.items()
        }
    assert [s.tool for s in memo.steps] == [
        "get_application",
        "get_documents_fields",
        "get_bureau_report",
        "get_cashflow_features",
        "run_decision",
        "get_reason_codes",
        "get_pricing",
        "search_policy",
    ]
    assert memo.mode == "demo" and memo.citations
    titles = [s.title for s in memo.sections]
    assert "Borçluluk" in titles and "Nakit Akışı" in titles
    assert facts == {}  # a fresh agent has collected nothing


async def test_live_agent_uses_tool_calls_then_guarded_memo(app_id):
    calls = [
        LLMResponse(tool_calls=[ToolCall(id="1", name="get_application", arguments={})]),
        LLMResponse(tool_calls=[ToolCall(id="2", name="run_decision", arguments={})]),
        LLMResponse(tool_calls=[ToolCall(id="3", name="draft_memo", arguments={})]),
        "## Başvuran Profili ve Talep\nTutar 999.999 TL [f:loan.amount].\n## Öneri\nOnay.",  # hallucinated number
        "## Başvuran Profili ve Talep\nTutar yine 999.999 TL [f:loan.amount].\n## Öneri\nOnay.",
    ]
    provider = DeterministicLLMProvider(calls)
    with session_scope() as session:
        application = session.get(Application, app_id)
        memo = await UnderwriterAgent(
            session, application, LLMService(LIVE, provider=provider, mode="live")
        ).run()
    assert [s.tool for s in memo.steps] == ["get_application", "run_decision"]
    assert (
        memo.mode == "fallback" and memo.error_kind == "citation"
    )  # guard rejected the invented figure
    assert memo.protocol == "tools"


async def test_agent_falls_back_to_react_protocol(app_id):
    def responder(messages):
        if any("Araç çağırma desteklenmiyor" in m.get("content", "") for m in messages):
            done = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("["))
            return "Eylem: get_pricing\nGirdi: {}" if done == 0 else "Eylem: draft_memo"
        if messages[-1]["content"].startswith("Aşağıdaki gözlemlerle"):
            return "## Öneri\nYetkili onayına sunulur; talep 180.000 TL [f:loan.amount]."
        return ToolsNotSupportedError()

    with session_scope() as session:
        application = session.get(Application, app_id)
        memo = await UnderwriterAgent(
            session,
            application,
            LLMService(LIVE, provider=DeterministicLLMProvider(responder), mode="live"),
        ).run()
    assert memo.protocol == "react"
    assert [s.tool for s in memo.steps] == ["get_pricing"]
    assert memo.recommendation.startswith("Yetkili")


def test_max_steps_constant():
    assert MAX_STEPS == 8


def test_memo_endpoint_and_audit(client, users, app_id):
    created = client.post(f"/api/v1/agent/{app_id}/memo", headers=users["uzman"])
    assert created.status_code == 201
    memo = created.json()
    assert "öneri niteliğindedir" in memo["disclaimer"]
    fetched = client.get(f"/api/v1/agent/{app_id}/memo", headers=users["uzman"]).json()
    assert fetched["application_id"] == app_id
    trail = client.get(f"/api/v1/applications/{app_id}/audit", headers=users["uzman"]).json()[
        "entries"
    ]
    assert sum(1 for e in trail if e["action"] == "AGENT_STEP") >= 8
    assert client.post(f"/api/v1/agent/{app_id}/memo", headers=users["basvuran"]).status_code == 403
    answer = client.post(
        "/api/v1/policy/ask",
        json={"question": "Yetki matrisi limitleri nelerdir?"},
        headers=users["uzman"],
    ).json()
    assert answer["sources"] and answer["cited"]


# ------------------------------------------------------------------ governance
def test_psi_and_drift():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25
    report = compute_drift([{"x": 0.9}] * 50, {"x": {"edges": [0, 0.5, 1], "shares": [0.5, 0.5]}})
    assert report["features"]["x"]["status"] == "ANLAMLI_KAYMA" and report["alerts"]


def test_model_inventory_card_and_promotion_four_eyes(client, users, app_id, monkeypatch):
    listing = client.get("/api/v1/models", headers=users["modelyon"]).json()["models"]
    ids = {m["model_id"]: m for m in listing}
    assert (
        ids["pd_lgbm_v2"]["role"] == "champion" and ids["challenger_lr_v2"]["role"] == "challenger"
    )
    card = client.get("/api/v1/models/pd_lgbm_v2/card", headers=users["modelyon"]).json()
    assert "cinsiyet" in card["excluded_attributes"][0] and card["metrics"]["auc"] > 0.7
    md = client.get("/api/v1/models/pd_lgbm_v2/card?format=md", headers=users["modelyon"])
    assert md.text.startswith("# Model Kartı")
    pdf = client.get("/api/v1/models/pd_lgbm_v2/card?format=pdf", headers=users["modelyon"])
    assert pdf.content.startswith(b"%PDF")
    cc = client.get("/api/v1/governance/champion-challenger", headers=users["modelyon"]).json()
    assert cc["decisions"] >= 1 and 0 <= cc["decision_agreement"] <= 1
    evidence = cc["validation"]
    assert evidence["available"] and evidence["delong_p_value"] < 0.05
    # On real data the LR challenger is significantly weaker: promotion is refused.
    blocked = client.post("/api/v1/models/challenger_lr_v2/promote", headers=users["modelyon"])
    assert blocked.status_code == 409 and "kanıt" in blocked.json()["detail"]
    from app.governance import inventory

    monkeypatch.setattr(
        inventory,
        "promotion_evidence",
        lambda champion, challenger: {"allowed": True, "reason": "test: kanıt yeterli"},
    )
    first = client.post("/api/v1/models/challenger_lr_v2/promote", headers=users["modelyon"]).json()
    assert first["role"] == "challenger"
    again = client.post("/api/v1/models/challenger_lr_v2/promote", headers=users["modelyon"])
    assert again.status_code == 403
    assert (
        client.post("/api/v1/models/challenger_lr_v2/promote", headers=users["uzman"]).status_code
        == 403
    )
    second = client.post(
        "/api/v1/models/challenger_lr_v2/promote", headers=users["modelyon2"]
    ).json()
    assert second["role"] == "champion" and second["status"] == "TERFI_ONAYLANDI"


def test_drift_fairness_endpoints(client, users):
    drift = client.get("/api/v1/governance/drift", headers=users["modelyon"]).json()
    assert drift["observations"] >= 1 and "dsr" in drift["features"]
    fairness = client.get("/api/v1/governance/fairness", headers=users["modelyon"]).json()
    assert fairness["live"]["decisions"] >= 1
    assert fairness["offline"] is None or fairness["offline"]["attributes"]["gender"]["min_air"] > 0
    real = fairness["real_data"]["uci_taiwan"]
    assert set(real["attributes"]) == {"SEX", "AGE_BAND", "EDUCATION", "MARRIAGE"}
    assert {r["approval_rate"] for r in real["lda"]["rows"]} == {real["approval_rate"]}


def test_validation_endpoint_is_for_model_managers(client, users):
    body = client.get("/api/v1/models/validation", headers=users["modelyon"]).json()
    taiwan = body["sets"]["uci_taiwan"]
    assert taiwan["holdout"]["lightgbm"]["auc_ci"][0] < taiwan["holdout"]["lightgbm"]["auc"]
    assert "delong" in taiwan and taiwan["champion"]["model"]
    assert "sentetik" in body["note"]
    assert client.get("/api/v1/models/validation", headers=users["uzman"]).status_code == 403


def test_rule_set_backtest_and_two_approvals(client, users):
    from app.core.rules import policy_file_text

    stricter = (
        policy_file_text()
        .replace("version: policy_v2", "version: policy_v3")
        .replace("auto_approve_max_pd: 0.08", "auto_approve_max_pd: 0.01")
    )
    backtest = client.post(
        "/api/v1/rule-sets/backtest",
        json={"version": "policy_v3", "content": stricter},
        headers=users["modelyon"],
    ).json()
    assert (
        backtest["decisions"] >= 1
        and backtest["approval_rate_after"] <= backtest["approval_rate_before"]
    )
    bad = client.post(
        "/api/v1/rule-sets",
        json={"version": "policy_v4", "content": stricter},
        headers=users["modelyon"],
    )
    assert bad.status_code == 422  # version mismatch
    created = client.post(
        "/api/v1/rule-sets",
        json={"version": "policy_v3", "content": stricter},
        headers=users["modelyon"],
    ).json()
    assert created["status"] == "TASLAK"
    assert (
        client.post("/api/v1/rule-sets/policy_v3/approve", headers=users["modelyon"]).json()[
            "status"
        ]
        == "TASLAK"
    )
    assert (
        client.post("/api/v1/rule-sets/policy_v3/approve", headers=users["modelyon"]).status_code
        == 403
    )
    active = client.post("/api/v1/rule-sets/policy_v3/approve", headers=users["komite"]).json()
    assert active["status"] == "YURURLUKTE"
    listing = client.get("/api/v1/rule-sets", headers=users["modelyon"]).json()
    assert any(
        r["version"] == "policy_v3" and r["status"] == "YURURLUKTE" for r in listing["rule_sets"]
    )
    # Deactivate again so later tests keep using policy_v2.
    with session_scope() as session:
        from app.db.models import RuleSet

        session.get(RuleSet, "policy_v3").status = "ARSIV"


def test_watchlist_after_disbursal(client, users):
    application_id = submit_complete(client, users["basvuran"], "temiz", requested_amount=90_000)
    client.post(f"/api/v1/applications/{application_id}/offer/accept", headers=users["basvuran"])
    assert (
        client.post(
            f"/api/v1/workbench/{application_id}/disburse", headers=users["uzman"]
        ).status_code
        == 200
    )
    watch = client.get("/api/v1/portfolio/watchlist", headers=users["uzman"]).json()
    entry = next(e for e in watch["entries"] if e["application_id"] == application_id)
    assert entry["months_observed"] == 6 and 0 <= entry["ews_score"] <= 1
    assert json.dumps(entry)
