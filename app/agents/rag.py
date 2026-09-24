"""Retrieval: per-application document index and the policy knowledge base.

* :class:`ApplicationDocumentIndex` — chunks of one application's documents
  only (isolated by ``application_id``; no cross-application leakage).
* :class:`PolicyKnowledgeBase` — ``docs/policies/*.md`` (own summaries of the
  internal credit policy and BDDK regulation). Ranking uses BM25
  (``rank_bm25``); when ``RAG_EMBEDDINGS=st`` and the optional
  ``sentence-transformers`` extra is available, a multilingual embedding
  model re-ranks the BM25 candidates.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.core.config import PROJECT_ROOT, get_settings
from app.documents.extraction import extract_text

POLICY_DIR = PROJECT_ROOT / "docs" / "policies"
_TR_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "İ": "i", "ö": "o", "ş": "s", "ü": "u"})


def tokenize(text: str) -> list[str]:
    lowered = text.replace("I", "ı").replace("İ", "i").lower().translate(_TR_FOLD)
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return [t[:6] for t in tokens if len(t) > 1]  # crude Turkish stemming by prefix


class Chunk(BaseModel):
    chunk_id: str
    source: str
    title: str
    text: str
    page: int | None = None


class _Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = (
            BM25Okapi([tokenize(c.text + " " + c.title) for c in chunks]) if chunks else None
        )

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        if not self._bm25 or not query.strip():
            return []
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        wanted = set(tokens)
        # BM25 IDF can be <= 0 in tiny corpora (one document); keep lexical matches anyway.
        overlap = [len(wanted & set(tokenize(c.text + " " + c.title))) for c in self.chunks]
        ranked = sorted(
            zip(scores, overlap, self.chunks, strict=True),
            key=lambda t: (t[1] > 0, t[0], t[1]),
            reverse=True,
        )
        hits = [(max(s, 0.0) + o, c) for s, o, c in ranked if s > 0 or o > 0][: max(k * 3, k)]
        hits = _maybe_rerank(query, hits)[:k]
        return [{**c.model_dump(), "score": round(float(s), 4)} for s, c in hits]


def _maybe_rerank(query: str, hits: list[tuple[float, Chunk]]) -> list[tuple[float, Chunk]]:
    if os.environ.get("RAG_EMBEDDINGS", "bm25") != "st" or not hits:
        return hits
    try:  # pragma: no cover - optional extra
        from sentence_transformers import util

        model = _embedder()
        q = model.encode(query, convert_to_tensor=True)
        docs = model.encode([c.text for _, c in hits], convert_to_tensor=True)
        sims = util.cos_sim(q, docs)[0].tolist()
        return sorted(
            ((float(sim), c) for sim, (_, c) in zip(sims, hits, strict=True)),
            key=lambda p: p[0],
            reverse=True,
        )
    except Exception:  # pragma: no cover
        return hits


@lru_cache(maxsize=1)
def _embedder() -> Any:  # pragma: no cover - optional extra
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


class _Splitter:
    """Paragraph → line → word packing into overlapping chunks (no external deps)."""

    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def split_text(self, text: str) -> list[str]:
        pieces = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            words = piece.split() if len(piece) > self.size else [piece]
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if len(candidate) > self.size and current:
                    chunks.append(current)
                    current = (
                        (current[-self.overlap :] + " " + word).strip() if self.overlap else word
                    )
                else:
                    current = candidate
        if current:
            chunks.append(current)
        return chunks


def _splitter() -> _Splitter:
    settings = get_settings()
    return _Splitter(settings.document_chunk_size, settings.document_chunk_overlap)


class ApplicationDocumentIndex(_Index):
    @classmethod
    def for_documents(
        cls, application_id: str, documents: Iterable[Any]
    ) -> ApplicationDocumentIndex:
        splitter = _splitter()
        chunks: list[Chunk] = []
        for document in documents:
            if document.application_id != application_id:
                continue  # hard isolation guard
            result = extract_text(Path(document.stored_path), document.mime)
            for page in result.pages:
                for i, piece in enumerate(splitter.split_text(page.text)):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.id}:{page.page}:{i}",
                            source=document.code,
                            title=document.filename,
                            text=piece,
                            page=page.page,
                        )
                    )
        return cls(chunks)


class PolicyKnowledgeBase(_Index):
    @classmethod
    def load(cls, directory: Path = POLICY_DIR) -> PolicyKnowledgeBase:
        splitter = _splitter()
        chunks: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            sections = re.split(r"\n(?=## )", text)
            doc_title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
            for s_index, section in enumerate(sections):
                heading = section.splitlines()[0].lstrip("# ").strip() if section else ""
                for c_index, piece in enumerate(splitter.split_text(section)):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{path.stem}#{s_index}.{c_index}",
                            source=path.name,
                            title=f"{doc_title} — {heading}",
                            text=piece,
                        )
                    )
        return cls(chunks)


@lru_cache(maxsize=1)
def policy_kb() -> PolicyKnowledgeBase:
    return PolicyKnowledgeBase.load()
