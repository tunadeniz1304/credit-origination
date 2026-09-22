"""LLM provider abstraction.

Real calls (OpenAI/Anthropic SDKs) are gated on the corresponding API key
being configured; without keys a deterministic :class:`MockLLMProvider` is
used, keeping every pipeline path offline-safe and testable. Prompt
chaining in the decision engine is expressed only through ``complete()``,
so a mock subclass can pin any step of the chain for deterministic tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.integrations.providers import stable_digest


class LLMProvider(ABC):
    """Portable chat-completion contract used by the prompt chain."""

    name: str

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return the model's completion for (system, user) messages."""


class OpenAIProvider(LLMProvider):
    """Real OpenAI chat completions via the official SDK."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._model = model
        self.logger = get_logger("llm.openai")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.logger.info("OpenAI completion requested (model=%s)", self._model)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()


class AnthropicProvider(LLMProvider):
    """Real Anthropic chat completions via the official SDK."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        from anthropic import Anthropic

        self._model = model
        self.logger = get_logger("llm.anthropic")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")
        self._client = Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.logger.info("Anthropic completion requested (model=%s)", self._model)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        parts = [block.text for block in message.content if getattr(block, "type", None) == "text"]
        return "".join(parts).strip()


class MockLLMProvider(LLMProvider):
    """Deterministic, offline-safe fallback.

    ``responses`` may be a dict keyed by the exact user prompt, or a callable
    ``(system, user) -> str`` for scripted test behaviour. The default reply is
    a stable digest of the user prompt, so repeated calls are reproducible.
    """

    name = "mock"

    def __init__(
        self,
        responses: dict[str, str] | Callable[[str, str], str] | None = None,
    ) -> None:
        self._responses = responses or {}
        self._calls: list[tuple[str, str]] = []

    @property
    def calls(self) -> list[tuple[str, str]]:
        """Every (system, user) pair completed so far (for assertions)."""
        return list(self._calls)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self._calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        if user in self._responses:
            return self._responses[user]
        return f"[mock-llm] deterministic reply #{stable_digest(user) % 10_000}"


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Resolve the active LLM provider (real only when a key is configured)."""
    settings = settings or get_settings()
    if settings.openai_api_key:
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.llm_model)
    if settings.anthropic_api_key:
        return AnthropicProvider(api_key=settings.anthropic_api_key)
    return MockLLMProvider()
