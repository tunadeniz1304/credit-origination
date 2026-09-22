"""RAG document analysis models."""
from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class DocumentChunk(BaseModel):
    """One chunked, retrievable unit of an applicant-supplied document."""

    source: str  # file path / document name
    text: str
    page: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        return len(self.text)


class RAGAnalysis(BaseModel):
    """Outcome of loading, chunking and indexing applicant documents."""

    chunks: list[DocumentChunk] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_hits: list[DocumentChunk] = Field(default_factory=list)  # top-k for a query
    query: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_chars(self) -> int:
        return sum(chunk.char_count for chunk in self.chunks)
