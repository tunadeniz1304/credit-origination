# ADR 0006 — OpenAI-compatible LLM client for data localisation

**Status:** accepted

## Context
The BDDK information-systems regulation requires primary systems in Turkey; sending customer data to foreign LLM APIs is problematic. The team currently uses DeepSeek V4 Flash through an OpenAI-compatible gateway.

## Decision
Use a single `AsyncOpenAI(api_key, base_url, timeout, max_retries)` client for Chat Completions. Switching to an on-premise model served by vLLM or Ollama is a one-line change (`LLM_BASE_URL`, `LLM_MODEL`). Keys resolve through aliases (`LLM_API_KEY` → `DEEPSEEK_API_KEY` → `GATEWAY_API_KEY` → `OPENAI_API_KEY`); `LLM_MODE=auto|live|demo`. Anthropic remains an optional provider. `reasoning_content` is ignored; empty answers caused by exhausted reasoning budgets are retried with a larger budget.

## Consequences
+ Vendor independence and a credible data-localisation story.
+ PII redaction makes even external calls pseudonymous.
− Features specific to one vendor are intentionally not used.
