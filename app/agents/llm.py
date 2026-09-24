"""LLM provider abstraction (OpenAI-compatible first).

Three providers share one async contract:

* :class:`OpenAICompatibleProvider` — the default. Any OpenAI-compatible Chat
  Completions endpoint (DeepSeek via a gateway, vLLM, Ollama, OpenAI) selected
  purely through ``LLM_BASE_URL`` / ``LLM_MODEL``; this is what makes BDDK data
  localisation a one-line configuration change.
* :class:`AnthropicProvider` — optional, model from ``ANTHROPIC_MODEL``.
* :class:`DeterministicLLMProvider` — demo mode / tests. It never touches the
  network; business narratives in demo mode are rendered by templates in
  :mod:`app.agents.narrator`, so this provider only needs to support scripted
  replies and tool-call sequences.

Providers raise :class:`LLMCallError` with a normalised ``kind`` so callers can
fall back per call. The LLM never makes binding credit decisions.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

LLMMode = Literal["live", "demo", "fallback"]
ErrorKind = Literal[
    "timeout",
    "rate_limit",
    "server_error",
    "connection",
    "bad_request",
    "invalid_json",
    "citation",
    "auth",
    "unknown",
]


class ToolCall(BaseModel):
    """One function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalised provider response (``reasoning_content`` is ignored)."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0


class LLMCallError(RuntimeError):
    """Provider call failed; ``kind`` drives fallback bookkeeping."""

    def __init__(self, kind: ErrorKind, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind: ErrorKind = kind


class ToolsNotSupportedError(LLMCallError):
    """The endpoint rejected the ``tools`` parameter (use the ReAct fallback)."""

    def __init__(self, message: str = "tools not supported") -> None:
        super().__init__("bad_request", message)


Message = dict[str, Any]


class LLMProvider(ABC):
    """Async chat-completion contract."""

    name: str = "abstract"
    model: str = ""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return the model response for ``messages``."""


def _classify_openai_error(exc: Exception) -> ErrorKind:
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return "auth"
    if isinstance(exc, openai.BadRequestError):
        return "bad_request"
    if isinstance(exc, openai.APIConnectionError):
        return "connection"
    if isinstance(exc, openai.APIStatusError):
        return "server_error" if exc.status_code >= 500 else "bad_request"
    return "unknown"


class OpenAICompatibleProvider(LLMProvider):
    """Chat Completions over the official ``openai`` SDK with a custom base URL."""

    name = "openai-compatible"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        from openai import AsyncOpenAI

        self.settings = settings
        self.model = settings.llm_model
        self.logger = get_logger("agent.llm")
        self._client = client or AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value() or "missing",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            if json_mode:
                # Endpoint does not support response_format: retry as plain text;
                # the caller extracts + validates JSON itself.
                kwargs.pop("response_format", None)
                try:
                    completion = await self._client.chat.completions.create(**kwargs)
                except Exception as inner:
                    raise LLMCallError(
                        _classify_openai_error(inner), type(inner).__name__
                    ) from inner
            elif tools:
                raise ToolsNotSupportedError(type(exc).__name__) from exc
            else:
                raise LLMCallError("bad_request", type(exc).__name__) from exc
        except Exception as exc:
            raise LLMCallError(_classify_openai_error(exc), type(exc).__name__) from exc

        first = completion.choices[0]
        if (
            not (first.message.content or "").strip()
            and getattr(first, "finish_reason", "") == "length"
        ):
            # Reasoning models (e.g. DeepSeek) can spend the whole budget on
            # reasoning_content; retry once with a larger completion budget.
            kwargs["max_tokens"] = int(kwargs["max_tokens"]) * 4
            try:
                completion = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                raise LLMCallError(_classify_openai_error(exc), type(exc).__name__) from exc
        latency = (time.perf_counter() - started) * 1000
        choice = completion.choices[0].message
        tool_calls: list[ToolCall] = []
        for call in getattr(choice, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except ValueError:
                arguments = {}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            content=(choice.content or "").strip(),  # reasoning_content deliberately ignored
            tool_calls=tool_calls,
            model=getattr(completion, "model", self.model) or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency,
        )


class AnthropicProvider(LLMProvider):
    """Optional Anthropic Messages provider (model configured via env)."""

    name = "anthropic"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        from anthropic import AsyncAnthropic

        self.settings = settings
        self.model = settings.anthropic_model
        self._client = client or AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value() or "missing",
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] in ("user", "assistant")]
        if tools:
            raise ToolsNotSupportedError("anthropic tool bridge not configured")
        started = time.perf_counter()
        try:
            message = await self._client.messages.create(
                model=self.model,
                system=system,
                messages=convo,  # type: ignore[arg-type]
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature if temperature is None else temperature,
            )
        except Exception as exc:
            raise LLMCallError("unknown", type(exc).__name__) from exc
        text = "".join(
            getattr(block, "text", "")
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        return LLMResponse(
            content=text.strip(),
            model=self.model,
            prompt_tokens=getattr(message.usage, "input_tokens", None),
            completion_tokens=getattr(message.usage, "output_tokens", None),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


ScriptedReply = str | LLMResponse | Exception
Responder = Callable[[Sequence[Message]], ScriptedReply]


class DeterministicLLMProvider(LLMProvider):
    """Offline provider for demo mode and tests.

    ``script`` is either a list of replies consumed in order or a callable
    ``messages -> reply``. A reply may be a string, a full :class:`LLMResponse`
    (e.g. with tool calls) or an exception instance to raise. With no script the
    provider echoes an empty JSON object so JSON callers can fall back cleanly.
    """

    name = "deterministic"
    model = "deterministic-demo"

    def __init__(self, script: Sequence[ScriptedReply] | Responder | None = None) -> None:
        self._script = script
        self._index = 0
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        reply: ScriptedReply
        if callable(self._script):
            reply = self._script(messages)
        elif self._script:
            reply = self._script[min(self._index, len(self._script) - 1)]
            self._index += 1
        else:
            reply = "{}"
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(content=reply, model=self.model, prompt_tokens=0, completion_tokens=0)


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Instantiate the provider for the effective LLM mode."""
    settings = settings or get_settings()
    if settings.llm_effective_mode == "demo":
        return DeterministicLLMProvider()
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(settings)
    return OpenAICompatibleProvider(settings)
