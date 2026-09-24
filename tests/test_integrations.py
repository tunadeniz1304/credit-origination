"""Integration client tests: persona mocks, consent gate, retries, shared breaker."""

from __future__ import annotations

import fakeredis
import httpx
import pytest

from app.core.config import Settings
from app.integrations import personas
from app.integrations.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    MemoryBreakerStore,
    RedisBreakerStore,
    get_breaker,
    reset_registry,
)
from app.integrations.clients import (
    CONSENT_EDEVLET,
    CONSENT_KKB,
    CONSENT_OPEN_BANKING,
    ConsentRequiredError,
    GIBClient,
    KKBClient,
    OpenBankingClient,
    SGKClient,
)

ALL = [CONSENT_KKB, CONSENT_EDEVLET, CONSENT_OPEN_BANKING]
TCKN = personas.DEMO_TCKN["temiz"]


@pytest.fixture(autouse=True)
def _fresh_breakers():
    reset_registry(MemoryBreakerStore())
    yield
    reset_registry(MemoryBreakerStore())


async def test_kkb_mock_is_deterministic_and_persona_consistent():
    client = KKBClient(Settings(_env_file=None))
    first = await client.get_report(TCKN, income_hint=45_000, consents=ALL)
    second = await client.get_report(TCKN, income_hint=45_000, consents=ALL)
    assert first == second
    assert first["provider"].endswith("(MOCK)")
    assert first["score"] >= 1400 and first["delinquency_count_24m"] == 0
    thin = await client.get_report(
        personas.DEMO_TCKN["ince_dosya"], income_hint=35_000, consents=ALL
    )
    assert thin["bureau_hit"] is False and thin["score"] is None


async def test_sgk_gib_and_open_banking_mocks():
    settings = Settings(_env_file=None)
    sgk = await SGKClient(settings).get_record(
        TCKN, income_hint=45_000, employment_type="MAASLI", consents=ALL
    )
    assert sgk["employment_months"] >= 36 and len(sgk["reported_gross_earnings_12m"]) == 12
    gib = await GIBClient(settings).get_tax_record(TCKN, income_hint=45_000, consents=ALL)
    assert gib["registered"] is True
    ob = await OpenBankingClient(settings).get_transactions(TCKN, income_hint=45_000, consents=ALL)
    assert len(ob["transactions"]) > 100


async def test_consent_required_no_request_made():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    client = OpenBankingClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    with pytest.raises(ConsentRequiredError):
        await client.get_transactions(TCKN, income_hint=1, consents=[CONSENT_KKB])
    with pytest.raises(ConsentRequiredError):
        await client.get_transactions(TCKN, income_hint=1, consents=None)
    assert calls == []


async def test_retry_recovers_after_transient_failures():
    attempts = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"score": 1500, "bureau_hit": True})

    client = KKBClient(Settings(_env_file=None), transport=httpx.MockTransport(flaky))
    data = await client.get_report(TCKN, income_hint=1, consents=ALL)
    assert data["score"] == 1500 and attempts["n"] == 3


async def test_timeout_setting_reaches_httpx():
    client = KKBClient(Settings(_env_file=None, api_timeout_seconds=1.25))
    session = await client._session()
    assert session.timeout.read == 1.25  # bug #14 regression
    await client.close()


async def test_breaker_is_shared_across_client_instances():
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    settings = Settings(
        _env_file=None, circuit_failure_threshold=2, circuit_reset_timeout_seconds=60
    )
    for _ in range(2):
        client = KKBClient(
            settings, transport=httpx.MockTransport(down)
        )  # new instance per request
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_report(TCKN, income_hint=1, consents=ALL)
    fresh = KKBClient(settings, transport=httpx.MockTransport(down))
    assert fresh.breaker.state is CircuitState.OPEN  # bug #8 regression
    with pytest.raises(CircuitOpenError):
        await fresh.get_report(TCKN, income_hint=1, consents=ALL)


def test_redis_store_shares_state_between_breakers():
    client = fakeredis.FakeRedis()
    a = CircuitBreaker(
        "kkb", failure_threshold=1, reset_timeout=60, store=RedisBreakerStore(client)
    )
    b = CircuitBreaker(
        "kkb", failure_threshold=1, reset_timeout=60, store=RedisBreakerStore(client)
    )
    with pytest.raises(RuntimeError):
        a.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert b.state is CircuitState.OPEN
    b.reset()
    assert a.state is CircuitState.CLOSED


def test_get_breaker_returns_singleton():
    assert get_breaker("x") is get_breaker("x")


def test_fault_injection_makes_kkb_fail():
    settings = Settings(_env_file=None, fault_injection_rate=1.0)
    handler = __import__("app.integrations.clients", fromlist=["mock_handler"]).mock_handler(
        settings
    )
    response = handler(httpx.Request("GET", f"http://x/kkb/report/{TCKN}"))
    assert response.status_code == 503


def test_personas_distribution_and_pins():
    assert personas.persona_for(personas.DEMO_TCKN["gri"]).key == "gri"
    assert personas.persona_for(personas.DEMO_TCKN["halka"]).key == "temiz"
    keys = {personas.persona_for(f"{i:011d}").key for i in range(10_000_000_000, 10_000_000_300)}
    assert {"temiz", "gri"} <= keys


def test_statement_arithmetic_is_exact():
    ob = personas.open_banking_transactions(personas.DEMO_TCKN["gecikmeli"], 40_000)
    total = ob["account"]["opening_balance"] + sum(t["amount"] for t in ob["transactions"])
    assert round(total, 2) == round(ob["closing_balance"], 2)
