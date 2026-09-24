"""LLMService: mode resolution, redaction, validation, fallback and telemetry.

All narrative generation goes through :meth:`LLMService.generate`:

1. ``demo`` mode → the deterministic fallback renderer (``mode="demo"``).
2. ``live`` mode → redact PII, call the provider, parse/validate (JSON schema and
   an optional guard such as the citation check). One repair round-trip is
   attempted on invalid output; any remaining failure (timeout, 429, 5xx,
   invalid JSON, guard violation) yields the fallback with ``mode="fallback"``
   and ``error_kind`` set. LLM errors never fail an application.

Each call is recorded: in-memory counters (``/api/v1/llm/status``), Prometheus
metrics, a key-free log line and an optional persistent recorder (the
``llm_calls`` table).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.llm import (
    ErrorKind,
    LLMCallError,
    LLMMode,
    LLMProvider,
    LLMResponse,
    Message,
    build_provider,
)
from app.agents.redaction import Redactor
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

T = TypeVar("T", bound=BaseModel)
Guard = Callable[[Any], list[str]]


class LLMCallRecord(BaseModel):
    """One recorded call (no prompt text, no key)."""

    task: str
    mode: LLMMode
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_kind: str | None = None
    application_id: str | None = None


@dataclass
class LLMStats:
    """Process-wide counters surfaced by ``GET /api/v1/llm/status``."""

    calls: int = 0
    failures: int = 0
    fallbacks: int = 0
    last_latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_mode: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, rec: LLMCallRecord) -> None:
        with self._lock:
            self.calls += 1
            self.by_mode[rec.mode] = self.by_mode.get(rec.mode, 0) + 1
            if rec.mode == "fallback":
                self.fallbacks += 1
            if rec.error_kind:
                self.failures += 1
            if rec.mode != "demo":
                self.last_latency_ms = round(rec.latency_ms, 1)
            self.prompt_tokens += rec.prompt_tokens or 0
            self.completion_tokens += rec.completion_tokens or 0

    def reset(self) -> None:
        with self._lock:
            self.calls = self.failures = self.fallbacks = 0
            self.prompt_tokens = self.completion_tokens = 0
            self.last_latency_ms = None
            self.by_mode = {}


STATS = LLMStats()
_recorders: list[Callable[[LLMCallRecord], None]] = []


def register_recorder(fn: Callable[[LLMCallRecord], None]) -> None:
    """Attach a persistent recorder (e.g. the ``llm_calls`` table writer)."""
    if fn not in _recorders:
        _recorders.append(fn)


@dataclass
class Generation(Generic[T]):
    """Result of one generation task."""

    text: str
    mode: LLMMode
    parsed: T | None = None
    error_kind: ErrorKind | None = None
    model: str = ""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text (fenced or embedded)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except ValueError:
        match = _JSON_BLOCK_RE.search(cleaned)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


class LLMService:
    """Facade used by narrators and the underwriter agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        mode: LLMMode | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.mode: LLMMode = mode or self.settings.llm_effective_mode
        self.logger = get_logger("agent.llm")

    @property
    def model_name(self) -> str:
        return self.provider.model or self.settings.llm_model

    def _record(self, rec: LLMCallRecord) -> None:
        STATS.record(rec)
        try:
            from app.core.metrics import observe_llm_call

            observe_llm_call(rec)
        except Exception:  # metrics are best effort
            self.logger.debug("metrics update failed", exc_info=True)
        self.logger.info(
            "llm task=%s mode=%s model=%s latency_ms=%.0f tokens=%s/%s error=%s",
            rec.task,
            rec.mode,
            rec.model,
            rec.latency_ms,
            rec.prompt_tokens,
            rec.completion_tokens,
            rec.error_kind,
        )
        for recorder in list(_recorders):
            try:
                recorder(rec)
            except Exception:
                self.logger.warning("llm call recorder failed", exc_info=True)

    async def generate(
        self,
        task: str,
        messages: Sequence[Message],
        *,
        fallback: Callable[[], str],
        schema: type[T] | None = None,
        guard: Guard | None = None,
        redactor: Redactor | None = None,
        application_id: str | None = None,
    ) -> Generation[T]:
        """Run one narrative task with validation and per-call fallback."""
        if self.mode == "demo":
            text = fallback()
            parsed = schema.model_validate(extract_json(text)) if schema else None
            self._record(
                LLMCallRecord(
                    task=task,
                    mode="demo",
                    model="deterministic-demo",
                    latency_ms=0.0,
                    application_id=application_id,
                )
            )
            return Generation(text=text, mode="demo", parsed=parsed, model="deterministic-demo")

        redactor = redactor or Redactor()
        outbound: list[Message] = [
            {**m, "content": redactor.redact(str(m.get("content", "")))} for m in messages
        ]
        started = time.perf_counter()
        response: LLMResponse | None = None
        error_kind: ErrorKind | None = None
        for attempt in range(2):  # initial call + one repair
            try:
                response = await self.provider.chat(outbound, json_mode=schema is not None)
            except LLMCallError as exc:
                error_kind = exc.kind
                break
            restored = redactor.restore(response.content)
            problems, parsed_obj = self._validate(restored, schema, guard)
            if not problems:
                self._record(
                    LLMCallRecord(
                        task=task,
                        mode="live",
                        model=response.model or self.model_name,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        application_id=application_id,
                    )
                )
                return Generation(
                    text=restored, mode="live", parsed=parsed_obj, model=response.model
                )
            error_kind = "invalid_json" if problems[0].startswith("json:") else "citation"
            if attempt == 0:
                outbound = [
                    *outbound,
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": (
                            "Yanıtın doğrulamadan geçmedi: "
                            + "; ".join(problems[:5])
                            + ". Lütfen yalnızca bağlamdaki alanlara atıf yaparak ve "
                            "istenen biçimde yeniden yaz."
                        ),
                    },
                ]

        text = fallback()
        parsed = schema.model_validate(extract_json(text)) if schema else None
        self._record(
            LLMCallRecord(
                task=task,
                mode="fallback",
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                prompt_tokens=response.prompt_tokens if response else None,
                completion_tokens=response.completion_tokens if response else None,
                error_kind=error_kind or "unknown",
                application_id=application_id,
            )
        )
        return Generation(
            text=text,
            mode="fallback",
            parsed=parsed,
            error_kind=error_kind or "unknown",
            model=self.model_name,
        )

    @staticmethod
    def _validate(
        text: str, schema: type[T] | None, guard: Guard | None
    ) -> tuple[list[str], T | None]:
        parsed: T | None = None
        target: Any = text
        if schema is not None:
            try:
                parsed = schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as exc:
                return [f"json: {type(exc).__name__}"], None
            target = parsed
        if not text.strip():
            return ["json: empty response"], None
        problems = guard(target) if guard else []
        return problems, parsed


def llm_status(settings: Settings | None = None) -> dict[str, Any]:
    """Key-free status summary for the API and dashboard badge."""
    settings = settings or get_settings()
    return {
        "mode": settings.llm_effective_mode,
        "model": settings.llm_model
        if settings.llm_effective_mode == "live"
        else "deterministic-demo",
        "configured_model": settings.llm_model,
        "base_url_host": settings.llm_base_url_host,
        "key_present": settings.llm_key_present,
        "last_latency_ms": STATS.last_latency_ms,
        "calls": STATS.calls,
        "failures": STATS.failures,
        "fallbacks": STATS.fallbacks,
        "by_mode": dict(STATS.by_mode),
        "tokens": {"prompt": STATS.prompt_tokens, "completion": STATS.completion_tokens},
    }


def startup_banner(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.llm_effective_mode == "live":
        return f"LLM: CANLI ({settings.llm_model} @ {settings.llm_base_url_host})"
    reason = "LLM_MODE=demo" if settings.llm_mode == "demo" else "anahtar bulunamadı"
    return f"LLM: DEMO modu ({reason})"
