"""LLM contract tests: modes, aliases, live/fallback, JSON repair, citations, redaction.

No test touches the network: live calls are served by ``respx`` routes on a
fake base URL.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.agents.citations import validate_cited_text, validate_plain_text
from app.agents.llm import (
    DeterministicLLMProvider,
    LLMCallError,
    LLMResponse,
    OpenAICompatibleProvider,
    build_provider,
)
from app.agents.llm_service import STATS, LLMService, extract_json, llm_status, startup_banner
from app.agents.narrator import (
    AnalysisOutput,
    Fact,
    NarrativeContext,
    Narrator,
    ReasonText,
    render_analysis,
    render_committee_summary,
)
from app.agents.redaction import PIIMaskingFilter, Redactor, mask_text
from app.core.config import LLMConfigurationError, Settings, validate_llm_settings

BASE = "https://llm.test/v1"
TCKN = "10000000146"
NAME = "Ayşe Kaya"


def _settings(**overrides) -> Settings:
    base = {
        "llm_api_key": "fake-test-key-0000",
        "llm_base_url": BASE,
        "llm_mode": "live",
        "llm_max_retries": 0,
        "llm_timeout_seconds": 2,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _completion(content: str, **message_extra) -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, **message_extra},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _ctx(outcome: str = "OTOMATIK_RET") -> NarrativeContext:
    return NarrativeContext(
        application_id="APP-TEST00000001",
        applicant_name=NAME,
        outcome=outcome,  # type: ignore[arg-type]
        facts=[
            Fact(id="f:income.monthly", label="Gelir", value=45000.0, unit="TL"),
            Fact(id="f:loan.amount", label="Tutar", value=250000.0, unit="TL"),
            Fact(id="f:loan.term", label="Vade", value=36, unit="ay"),
            Fact(id="f:dsr", label="DSR", value=0.58, unit="%"),
            Fact(id="f:policy.max_dsr", label="Sınır", value=0.5, unit="%"),
            Fact(id="f:bureau.score", label="KKB", value=1320, unit="puan"),
            Fact(id="f:model.pd", label="PD", value=0.083, unit="%"),
        ],
        reason_codes=[
            ReasonText(
                code="R01_DSR_YUKSEK",
                text="Aylık borç ödemelerinizin gelirinize oranı %58 ile sınır olan %50'nin üzerinde.",
            )
        ],
        counterfactuals=["Vadeyi 48 aya çıkarmanız halinde onay olasılığı yükselir."],
        redactor_fields={"identity_no": TCKN, "phone": "05321234567"},
    )


# ---------------------------------------------------------------- settings
def test_key_aliases_first_match_wins(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    settings = Settings(_env_file=None)
    assert settings.llm_api_key.get_secret_value() == "fake-deepseek-key"
    assert settings.llm_base_url == "https://api.deepseek.com"  # generic public default
    assert settings.llm_model == "deepseek-v4-flash"


def test_auto_mode_resolves_on_key_presence(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "auto")
    monkeypatch.setenv("LLM_API_KEY", "")
    assert Settings(_env_file=None).llm_effective_mode == "demo"
    monkeypatch.setenv("LLM_API_KEY", "fake-live-key")
    assert Settings(_env_file=None).llm_effective_mode == "live"
    monkeypatch.setenv("LLM_MODE", "demo")
    assert Settings(_env_file=None).llm_effective_mode == "demo"


def test_forced_live_without_key_fails_fast():
    settings = Settings(_env_file=None, llm_mode="live", llm_api_key="")
    with pytest.raises(LLMConfigurationError):
        validate_llm_settings(settings)


def test_startup_banner_and_status_never_expose_key():
    live = _settings()
    assert startup_banner(live) == "LLM: CANLI (deepseek-v4-flash @ llm.test)"
    status = llm_status(live)
    assert status["key_present"] is True
    assert "fake-test" not in json.dumps(status)
    assert "DEMO" in startup_banner(Settings(_env_file=None, llm_mode="demo"))


def test_build_provider_by_mode():
    assert isinstance(
        build_provider(Settings(_env_file=None, llm_mode="demo")), DeterministicLLMProvider
    )
    assert isinstance(build_provider(_settings()), OpenAICompatibleProvider)


# ---------------------------------------------------------------- live path
@respx.mock
async def test_live_call_uses_base_url_and_ignores_reasoning_content():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_completion("Merhaba", reasoning_content="gizli düşünce")
        )
    )
    provider = OpenAICompatibleProvider(_settings(llm_temperature=0.3))
    response = await provider.chat([{"role": "user", "content": "selam"}])
    assert response.content == "Merhaba"
    assert response.prompt_tokens == 11
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "deepseek-v4-flash"
    assert sent["temperature"] == 0.3  # LLM_TEMPERATURE honoured
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-test-key-0000"


@respx.mock
async def test_json_mode_falls_back_to_plain_text_on_400():
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "response_format unsupported"}}),
            httpx.Response(200, json=_completion('```json\n{"a": 1}\n```')),
        ]
    )
    provider = OpenAICompatibleProvider(_settings())
    response = await provider.chat([{"role": "user", "content": "x"}], json_mode=True)
    assert extract_json(response.content) == {"a": 1}
    assert "response_format" in json.loads(route.calls[0].request.content)
    assert "response_format" not in json.loads(route.calls[1].request.content)


@pytest.mark.parametrize(
    ("status", "kind"), [(429, "rate_limit"), (500, "server_error"), (503, "server_error")]
)
@respx.mock
async def test_http_errors_are_classified(status, kind):
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(status, json={}))
    with pytest.raises(LLMCallError) as err:
        await OpenAICompatibleProvider(_settings()).chat([{"role": "user", "content": "x"}])
    assert err.value.kind == kind


@respx.mock
async def test_timeout_falls_back_and_marks_result():
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    service = LLMService(_settings())
    result = await service.generate(
        "committee_summary",
        [{"role": "user", "content": "x"}],
        fallback=lambda: "Deterministik özet.",
    )
    assert result.mode == "fallback"
    assert result.error_kind == "timeout"
    assert result.text == "Deterministik özet."


async def test_invalid_json_gets_one_repair_then_fallback():
    provider = DeterministicLLMProvider(["bu json değil", "hala değil"])
    service = LLMService(_settings(), provider=provider, mode="live")
    result = await service.generate(
        "analysis",
        [{"role": "user", "content": "x"}],
        fallback=lambda: render_analysis(_ctx()).model_dump_json(),
        schema=AnalysisOutput,
    )
    assert len(provider.calls) == 2  # original + repair
    assert "doğrulamadan geçmedi" in provider.calls[1][-1]["content"]
    assert result.mode == "fallback" and result.error_kind == "invalid_json"
    assert isinstance(result.parsed, AnalysisOutput)


async def test_repair_succeeds_on_second_attempt():
    good = json.dumps(
        {
            "summary": "Özet",
            "strengths": [{"text": "KKB notu 1.320 puan.", "field_ids": ["f:bureau.score"]}],
            "risks": [],
            "recommendation": "Ret",
        },
        ensure_ascii=False,
    )
    provider = DeterministicLLMProvider(["{bozuk", good])
    narrator = Narrator(LLMService(_settings(), provider=provider, mode="live"))
    result = await narrator.analysis(_ctx())
    assert result.mode == "live"
    assert result.parsed.strengths[0].field_ids == ["f:bureau.score"]


# ---------------------------------------------------------------- citation guard
def test_citation_guard_accepts_matching_numbers():
    ctx = _ctx()
    text = "DSR %58 [f:dsr] olup sınır %50 [f:policy.max_dsr]; tutar 250.000 TL [f:loan.amount]."
    assert validate_cited_text(text, ctx.fact_map()).ok


def test_citation_guard_rejects_unknown_field_and_wrong_number():
    facts = _ctx().fact_map()
    assert not validate_cited_text("Gelir 45.000 TL [f:income.unknown].", facts).ok
    wrong = validate_cited_text("Gelir 99.000 TL [f:income.monthly].", facts)
    assert not wrong.ok and "99.000" in wrong.errors[0]
    assert not validate_cited_text("Hiç atıf yok 45.000 TL.", facts).ok


async def test_hallucinated_number_triggers_repair_then_fallback():
    bad = "Gelir 99.999 TL [f:income.monthly] ve DSR %58 [f:dsr]."
    provider = DeterministicLLMProvider([bad, bad])
    narrator = Narrator(LLMService(_settings(), provider=provider, mode="live"))
    ctx = _ctx()
    analysis = render_analysis(ctx)
    result = await narrator.committee_summary(ctx, analysis)
    assert result.mode == "fallback" and result.error_kind == "citation"
    assert validate_cited_text(result.text, ctx.fact_map()).ok


def test_demo_templates_pass_their_own_guards():
    ctx = _ctx()
    summary = render_committee_summary(ctx, render_analysis(ctx))
    assert validate_cited_text(summary, ctx.fact_map()).ok, summary


# ---------------------------------------------------------------- chain
async def test_chain_step_two_receives_step_one_output():
    step1 = json.dumps(
        {
            "summary": "ÖZET-ADIM-1",
            "strengths": [{"text": "KKB notu 1.320 puan.", "field_ids": ["f:bureau.score"]}],
            "risks": [{"text": "DSR %58.", "field_ids": ["f:dsr"]}],
            "recommendation": "Ret",
        },
        ensure_ascii=False,
    )
    step2 = "DSR %58 [f:dsr] sınırın üzerinde."
    step3 = (
        "Sayın başvuran, başvurunuz olumsuz sonuçlanmıştır. KVKK kapsamında itiraz "
        "hakkınız bulunmaktadır."
    )
    provider = DeterministicLLMProvider([step1, step2, step3])
    bundle = await Narrator(LLMService(_settings(), provider=provider, mode="live")).run_chain(
        _ctx()
    )
    assert "ÖZET-ADIM-1" in provider.calls[1][-1]["content"]
    assert bundle.modes == {
        "analysis": "live",
        "committee_summary": "live",
        "applicant_letter": "live",
    }
    assert bundle.overall_mode == "live"


async def test_demo_chain_is_professional_turkish_without_placeholders():
    bundle = await Narrator(LLMService(Settings(_env_file=None, llm_mode="demo"))).run_chain(_ctx())
    text = bundle.committee_summary + bundle.applicant_letter
    assert "[mock-llm]" not in text
    assert "KREDİ KOMİTESİ ÖZETİ" in bundle.committee_summary
    assert "itiraz" in bundle.applicant_letter.lower()
    assert "6698 sayılı" in bundle.applicant_letter
    assert bundle.overall_mode == "demo"


async def test_llm_failure_never_raises_from_chain():
    provider = DeterministicLLMProvider([LLMCallError("server_error")] * 3)
    bundle = await Narrator(LLMService(_settings(), provider=provider, mode="live")).run_chain(
        _ctx()
    )
    assert bundle.overall_mode == "fallback"
    assert bundle.errors["analysis"] == "server_error"


# ---------------------------------------------------------------- redaction
def test_redactor_round_trip():
    redactor = Redactor.for_applicant(
        name=NAME, identity_no=TCKN, phone="0532 123 45 67", iban="TR330006100519786457841326"
    )
    raw = f"{NAME} ({TCKN}) tel 0532 123 45 67 IBAN TR330006100519786457841326 e-posta a@b.com"
    red = redactor.redact(raw)
    for secret in (NAME, TCKN, "0532 123 45 67", "TR330006100519786457841326", "a@b.com"):
        assert secret not in red
    assert "BASVURAN_1" in red and "IBAN_…1326" in red
    assert redactor.restore(red) == raw


@respx.mock
async def test_live_requests_contain_no_pii():
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(200, json=_completion("Sayın BASVURAN_1, itiraz hakkınız vardır."))

    respx.post(f"{BASE}/chat/completions").mock(side_effect=handler)
    narrator = Narrator(LLMService(_settings()))
    result = await narrator.applicant_letter(_ctx())
    body = b"".join(captured).decode("utf-8")
    assert TCKN not in body and NAME not in body and "05321234567" not in body
    assert "\\u015f" not in body or NAME.encode("unicode_escape").decode() not in body
    assert NAME in result.text  # pseudonym restored for the applicant


def test_log_filter_masks_pii(caplog):
    import logging

    assert TCKN not in mask_text(f"tckn={TCKN}")
    logger = logging.getLogger("test.pii")
    logger.addFilter(PIIMaskingFilter())
    with caplog.at_level(logging.INFO, logger="test.pii"):
        logger.info("başvuran %s iban %s", TCKN, "TR330006100519786457841326")
    assert TCKN not in caplog.text
    assert "TR330006100519786457841326" not in caplog.text


async def test_stats_track_calls_and_failures():
    STATS.reset()
    provider = DeterministicLLMProvider([LLMCallError("rate_limit")])
    await LLMService(_settings(), provider=provider, mode="live").generate(
        "x", [{"role": "user", "content": "y"}], fallback=lambda: "z"
    )
    assert STATS.calls == 1 and STATS.failures == 1 and STATS.fallbacks == 1


async def test_deterministic_provider_scripts_tool_calls():
    reply = LLMResponse(
        content="", tool_calls=[{"id": "1", "name": "get_application", "arguments": {}}]
    )
    provider = DeterministicLLMProvider([reply])
    response = await provider.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert response.tool_calls[0].name == "get_application"


def test_plain_text_guard():
    facts = _ctx().fact_map()
    assert validate_plain_text("Tutar 250.000 TL, vade 36 ay.", facts).ok
    assert not validate_plain_text("Tutar 275.000 TL.", facts).ok
