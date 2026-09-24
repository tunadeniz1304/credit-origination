"""Policy & regulation assistant ("Politika sor") with source citations.

Retrieves the most relevant chunks from ``docs/policies`` (BM25, optional
embeddings) and answers with ``[kaynak:<chunk_id>]`` citations. In live mode
the LLM writes the answer and the guard rejects citations to chunks that were
not retrieved; in demo mode (or on fallback) an extractive answer is composed
from the retrieved passages.
"""

from __future__ import annotations

import re
from typing import Any

from app.agents.llm_service import LLMService
from app.agents.rag import policy_kb

SOURCE_RE = re.compile(r"\[kaynak:([^\]]+)\]")


def _sentences(text: str, limit: int = 2) -> str:
    clean = re.sub(r"^#+\s.*$", "", text, flags=re.MULTILINE).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return " ".join(p for p in parts[:limit] if p)


def extractive_answer(question: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "Bu soruya politika belgelerinde doğrudan karşılık bulunamadı; lütfen kredi politikası birimine danışın."
    lines = [f"'{question}' sorusuyla ilgili politika hükümleri:"]
    for hit in hits[:3]:
        lines.append(f"- {_sentences(hit['text'])} [kaynak:{hit['chunk_id']}]")
    return "\n".join(lines)


async def ask(question: str, llm: LLMService | None = None, k: int = 4) -> dict[str, Any]:
    llm = llm or LLMService()
    hits = policy_kb().search(question, k=k)
    allowed = {h["chunk_id"] for h in hits}
    passages = "\n\n".join(f"[kaynak:{h['chunk_id']}] {h['title']}\n{h['text']}" for h in hits)

    def guard(text: str) -> list[str]:
        cited = set(SOURCE_RE.findall(text))
        errors = [f"unknown source {c}" for c in cited - allowed]
        if hits and not cited:
            errors.append("no sources cited")
        return errors

    generation: Any = await llm.generate(
        "policy_answer",
        [
            {
                "role": "system",
                "content": "Sen bir bankanın kredi politikası asistanısın. Yalnızca verilen pasajlara dayanarak "
                "kısa ve net Türkçe yanıt ver; her iddianın sonuna [kaynak:<id>] ekle. Pasajlarda yoksa bilmediğini söyle.",
            },
            {"role": "user", "content": f"Soru: {question}\n\nPasajlar:\n{passages}"},
        ],
        fallback=lambda: extractive_answer(question, hits),
        guard=guard,
    )
    return {
        "question": question,
        "answer": generation.text,
        "sources": [
            {
                "chunk_id": h["chunk_id"],
                "title": h["title"],
                "source": h["source"],
                "score": h["score"],
            }
            for h in hits
        ],
        "cited": sorted(set(SOURCE_RE.findall(generation.text))),
        "mode": generation.mode,
    }
