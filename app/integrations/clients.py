"""Resilient external API clients (KKB credit bureau / e-Devlet).

Every call goes through a real ``httpx`` session wrapped by Tenacity retry
(exponential backoff on transient failures) and a circuit breaker. By
default the session runs over an in-process ``httpx.MockTransport`` that
serves deterministic payloads (see ``providers``), so local development,
tests and Docker need no external service; set ``mock_external=false`` and a
live ``KKB_BASE_URL`` / ``EDEVLET_BASE_URL`` to hit real endpoints.
"""

from __future__ import annotations

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
from app.integrations.circuit_breaker import CircuitBreaker
from app.integrations.providers import derive_employment_record, derive_kbb_report
from app.models import EmploymentRecord, KBBReport

_RETRY_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException)


def _is_retryable(exception: BaseException) -> bool:
    if isinstance(exception, _RETRY_EXCEPTIONS):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return False


class ExternalClientError(RuntimeError):
    """Raised when an external provider cannot be satisfied."""


def _mock_kkb_handler(request: httpx.Request) -> httpx.Response:
    identity_no = request.url.path.rstrip("/").rsplit("/", 1)[-1]
    report = derive_kbb_report(identity_no)
    return httpx.Response(
        200,
        json={
            "identity_no": identity_no,
            "score": report.score,
            "risk_class": report.risk_class,
            "total_debt": report.total_debt,
        },
    )


def _mock_edevlet_handler(request: httpx.Request) -> httpx.Response:
    identity_no = request.url.path.rstrip("/").rsplit("/", 1)[-1]
    record = derive_employment_record(identity_no)
    return httpx.Response(
        200,
        json={
            "identity_no": identity_no,
            "employer": record.employer,
            "years": record.years,
            "verified": record.verified,
        },
    )


class ExternalAPIClient:
    """Shared driver: session lifecycle + retry + circuit breaker."""

    service: str = "external"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        mock_external: bool | None = None,
        failure_threshold: int | None = None,
        reset_timeout: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger(f"api.{self.service}")
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.timeout = timeout if timeout is not None else self.settings.api_timeout_seconds
        use_mock = self.settings.mock_external if mock_external is None else mock_external
        if transport is None and use_mock:
            transport = httpx.MockTransport(self._default_handler())
        self._transport = transport
        self.breaker = CircuitBreaker(
            name=self.service,
            failure_threshold=failure_threshold or self.settings.circuit_failure_threshold,
            reset_timeout=reset_timeout
            if reset_timeout is not None
            else self.settings.circuit_reset_timeout_seconds,
        )
        self._client: httpx.AsyncClient | None = None

    def _default_base_url(self) -> str:
        raise NotImplementedError

    def _default_handler(self) -> httpx.RequestHandler:
        raise NotImplementedError

    async def _session(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=1.5),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=before_sleep_log(get_logger("api.retry"), 20),
    )
    async def _get_json(self, path: str) -> dict:
        client = await self._session()
        response = await client.get(path)
        response.raise_for_status()
        return response.json()


class KKBClient(ExternalAPIClient):
    """Resilient KKB credit-bureau client."""

    service = "kkb"

    def _default_base_url(self) -> str:
        return self.settings.kkb_base_url

    def _default_handler(self) -> httpx.RequestHandler:
        return _mock_kkb_handler

    async def get_report(self, identity_no: str) -> KBBReport:
        async def _call() -> KBBReport:
            data = await self._get_json(f"/kkb/report/{identity_no}")
            return KBBReport(
                score=int(data["score"]),
                risk_class=data["risk_class"],
                total_debt=float(data["total_debt"]),
            )

        report = await self.breaker.call_async(_call)
        self.logger.info(
            "KKB report: identity=%s score=%d risk='%s' debt=%.2f",
            identity_no,
            report.score,
            report.risk_class,
            report.total_debt,
        )
        return report


class EDevletClient(ExternalAPIClient):
    """Resilient e-Devlet employment-record client."""

    service = "edevlet"

    def _default_base_url(self) -> str:
        return self.settings.edevlet_base_url

    def _default_handler(self) -> httpx.RequestHandler:
        return _mock_edevlet_handler

    async def get_employment(self, identity_no: str) -> EmploymentRecord:
        async def _call() -> EmploymentRecord:
            data = await self._get_json(f"/edevlet/employment/{identity_no}")
            return EmploymentRecord(
                employer=data["employer"],
                years=int(data["years"]),
                verified=bool(data["verified"]),
            )

        record = await self.breaker.call_async(_call)
        self.logger.info(
            "e-Devlet record: identity=%s employer='%s' years=%d verified=%s",
            identity_no,
            record.employer,
            record.years,
            record.verified,
        )
        return record
