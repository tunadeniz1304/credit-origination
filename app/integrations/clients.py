"""Resilient external data clients (all MOCK endpoints, clearly labelled).

* KKB / Findeks — credit score, facilities, arrears, inquiries, BBE index
* e-Devlet / SGK — employer, start date, premium days, reported earnings
* GİB — tax registration for self-employed applicants
* Open banking (ÖHVPS / BKM GEÇİT style) — 12 months of transactions

Every call runs over ``httpx`` with the configured timeout, Tenacity retries
on transient failures and the process-wide circuit breaker for the service.
By default an in-process ``httpx.MockTransport`` serves persona-consistent
payloads. Every call requires a matching, unexpired consent: without it no
request is made (:class:`ConsentRequiredError`).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.integrations import personas
from app.integrations.circuit_breaker import CircuitBreaker, get_breaker

_RETRY_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException)

CONSENT_KKB = "KKB_SORGU"
CONSENT_EDEVLET = "EDEVLET_SORGU"
CONSENT_OPEN_BANKING = "ACIK_BANKACILIK"
CONSENT_GIB = "EDEVLET_SORGU"


class ExternalClientError(RuntimeError):
    """Raised when an external provider cannot be satisfied."""


class ConsentRequiredError(PermissionError):
    """No valid consent for this data source: the call is not made."""


def _is_retryable(exception: BaseException) -> bool:
    if isinstance(exception, _RETRY_EXCEPTIONS):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return False


def _counts_against_service(exception: Exception) -> bool:
    """Circuit breaker verdict: a 4xx is the caller's problem, not an outage of the service."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return True


def _income(request: httpx.Request) -> float:
    try:
        return float(request.url.params.get("income_hint", "40000"))
    except ValueError:
        return 40000.0


def _identity(request: httpx.Request) -> str:
    return request.url.path.rstrip("/").rsplit("/", 1)[-1]


def mock_handler(settings: Settings) -> Any:
    """Single mock transport handler routing every provider path."""
    chaos = random.Random(7)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        tckn = _identity(request)
        income = _income(request)
        if "/kkb/" in path:
            if settings.fault_injection_rate and chaos.random() < settings.fault_injection_rate:
                return httpx.Response(503, json={"error": "injected fault"})
            return httpx.Response(200, json=personas.kkb_report(tckn, income))
        if "/sgk/" in path:
            employment = request.url.params.get("employment_type", "MAASLI")
            return httpx.Response(200, json=personas.sgk_record(tckn, income, employment))
        if "/gib/" in path:
            return httpx.Response(200, json=personas.gib_record(tckn, income))
        if "/openbanking/" in path:
            return httpx.Response(200, json=personas.open_banking_transactions(tckn, income))
        if "/barcode/" in path:
            from app.documents.samples import verify_barcode

            return httpx.Response(200, json={"valid": verify_barcode(tckn)})
        return httpx.Response(404, json={"error": "unknown mock path"})

    return handler


class ExternalAPIClient:
    """Shared driver: session lifecycle + timeout + retry + circuit breaker."""

    service: str = "external"
    consent_type: str | None = None

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        mock_external: bool | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(f"api.{self.service}")
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.timeout = timeout if timeout is not None else self.settings.api_timeout_seconds
        use_mock = self.settings.mock_external if mock_external is None else mock_external
        if transport is None and use_mock:
            transport = httpx.MockTransport(mock_handler(self.settings))
        self._transport = transport
        self.breaker = breaker or get_breaker(
            self.service,
            self.settings.circuit_failure_threshold,
            self.settings.circuit_reset_timeout_seconds,
        )
        self._client: httpx.AsyncClient | None = None

    def _default_base_url(self) -> str:
        raise NotImplementedError

    def _check_consent(self, consents: Iterable[str] | None) -> None:
        if self.consent_type is None:
            return
        if consents is None or self.consent_type not in set(consents):
            raise ConsentRequiredError(f"{self.service}: consent {self.consent_type} missing")

    async def _session(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                transport=self._transport,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=1.0),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=before_sleep_log(get_logger("api.retry"), 20),
    )
    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        client = await self._session()
        response = await client.get(path, params=params)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    async def fetch(
        self, path: str, *, consents: Iterable[str] | None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self._check_consent(consents)

        async def _call() -> dict[str, Any]:
            return await self._get_json(path, params)

        data: dict[str, Any] = await self.breaker.call_async(
            _call, is_failure=_counts_against_service
        )
        return data


class KKBClient(ExternalAPIClient):
    service = "kkb"
    consent_type = CONSENT_KKB

    def _default_base_url(self) -> str:
        return self.settings.kkb_base_url

    async def get_report(
        self, tckn: str, *, income_hint: float, consents: Iterable[str] | None
    ) -> dict[str, Any]:
        data = await self.fetch(
            f"/kkb/report/{tckn}", consents=consents, params={"income_hint": income_hint}
        )
        self.logger.info(
            "KKB report received: score=%s hit=%s", data.get("score"), data.get("bureau_hit")
        )
        return data


class SGKClient(ExternalAPIClient):
    service = "edevlet"
    consent_type = CONSENT_EDEVLET

    def _default_base_url(self) -> str:
        return self.settings.edevlet_base_url

    async def get_record(
        self,
        tckn: str,
        *,
        income_hint: float,
        employment_type: str,
        consents: Iterable[str] | None,
    ) -> dict[str, Any]:
        return await self.fetch(
            f"/sgk/record/{tckn}",
            consents=consents,
            params={"income_hint": income_hint, "employment_type": employment_type},
        )

    async def verify_barcode(self, barcode: str, *, consents: Iterable[str] | None) -> bool:
        data = await self.fetch(f"/barcode/verify/{barcode}", consents=consents)
        return bool(data.get("valid"))


class GIBClient(ExternalAPIClient):
    service = "gib"
    consent_type = CONSENT_GIB

    def _default_base_url(self) -> str:
        return self.settings.gib_base_url

    async def get_tax_record(
        self, tckn: str, *, income_hint: float, consents: Iterable[str] | None
    ) -> dict[str, Any]:
        return await self.fetch(
            f"/gib/taxpayer/{tckn}", consents=consents, params={"income_hint": income_hint}
        )


class OpenBankingClient(ExternalAPIClient):
    service = "openbanking"
    consent_type = CONSENT_OPEN_BANKING

    def _default_base_url(self) -> str:
        return self.settings.openbanking_base_url

    async def get_transactions(
        self, tckn: str, *, income_hint: float, consents: Iterable[str] | None
    ) -> dict[str, Any]:
        return await self.fetch(
            f"/openbanking/accounts/{tckn}", consents=consents, params={"income_hint": income_hint}
        )
