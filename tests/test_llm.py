"""LLMProvider contract tests (mock path — deterministic and offline)."""
from __future__ import annotations

from app.agents.llm import LLMProvider, MockLLMProvider, get_provider
from app.core.config import Settings


def test_mock_provider_is_deterministic():
    provider = MockLLMProvider()
    first = provider.complete("sys", "user prompt")
    second = provider.complete("sys", "user prompt")
    assert first == second
    assert ("sys", "user prompt") in provider.calls


def test_mock_provider_canned_response():
    provider = MockLLMProvider(responses={"HELLO": "world"})
    assert provider.complete("", "HELLO") == "world"


def test_get_provider_uses_mock_when_no_keys():
    settings = Settings(openai_api_key="", anthropic_api_key="")
    provider = get_provider(settings)
    assert isinstance(provider, LLMProvider)
    assert provider.name == "mock"
