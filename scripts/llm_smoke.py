"""One real LLM call when a key is configured; otherwise report demo mode.

Prints ``OK model=<model> latency=<ms>ms`` (exit 0) or ``DEMO modu`` (exit 0);
exits 1 when a live call fails. The API key is never printed.

Usage: python scripts/llm_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.llm import LLMCallError, OpenAICompatibleProvider
from app.core.config import get_settings


async def _call() -> int:
    settings = get_settings()
    if settings.llm_effective_mode != "live":
        print("DEMO modu (anahtar bulunamadı veya LLM_MODE=demo)")
        return 0
    provider = OpenAICompatibleProvider(settings)
    started = time.perf_counter()
    try:
        response = await provider.chat(
            [{"role": "user", "content": "Tek kelimeyle yanıt ver: Türkiye'nin başkenti?"}],
            max_tokens=32,
        )
    except LLMCallError as exc:
        print(f"FAIL kind={exc.kind} host={settings.llm_base_url_host}")
        return 1
    latency = (time.perf_counter() - started) * 1000
    print(
        f"OK model={response.model or settings.llm_model} latency={latency:.0f}ms reply={response.content[:40]!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_call()))
