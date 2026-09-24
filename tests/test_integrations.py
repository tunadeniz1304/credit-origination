"""External integration client tests: deterministic payloads, retry, breaker.

Each client is exercised over an in-process ``httpx.MockTransport``, so no
network is required and all behaviour (retry backoff, circuit trip) is fully
deterministic.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.integrations.circuit_breaker import CircuitOpenError
from app.integrations.clients import EDevletClient, KKBClient
from app.integrations.providers import (
    EMPLOYERS,
    derive_employment_record,
    derive_kbb_report,
)


def test_kkb_deterministic_default_transport():
    async def run() -> None:
        client = KKBClient()  # default in-process mock transport
        try:
            report = await client.get_report("12345678901")
            assert report.score == 1450
            assert report.total_debt == 134070.0
            assert report.risk_class == "DÜŞÜK RİSK"
        finally:
            await client.close()

    asyncio.run(run())


def test_edevlet_deterministic_default_transport():
    async def run() -> None:
        client = EDevletClient()
        try:
            record = await client.get_employment("12345678901")
            expected = derive_employment_record("12345678901")
            assert record == expected
            assert record.employer in EMPLOYERS
            assert 0 <= record.years < 35
            assert isinstance(record.verified, bool)
        finally:
            await client.close()

    asyncio.run(run())


def test_retry_recovers_after_transient_failures():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.TransportError("transient network blip")
        report = derive_kbb_report("12345678901")
        return httpx.Response(
            200,
            json={
                "score": report.score,
                "risk_class": report.risk_class,
                "total_debt": report.total_debt,
            },
        )

    async def run() -> None:
        client = KKBClient(base_url="http://mock", transport=httpx.MockTransport(handler))
        try:
            report = await client.get_report("12345678901")
            assert report.score == 1450
        finally:
            await client.close()

    asyncio.run(run())
    assert attempts["count"] == 3  # two retries after the first failure


def test_circuit_opens_when_provider_keeps_returning_5xx():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service unavailable"})

    async def run() -> None:
        client = KKBClient(
            base_url="http://mock",
            transport=httpx.MockTransport(handler),
            failure_threshold=2,
            reset_timeout=30.0,
        )
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_report("12345678901")
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_report("12345678901")
            # Breaker is now OPEN: hard rejection without hitting the provider.
            with pytest.raises(CircuitOpenError):
                await client.get_report("12345678901")
        finally:
            await client.close()

    asyncio.run(run())
