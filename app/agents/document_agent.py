"""Belge Kontrol Ajanı + RAG document analysis.

DocumentControlAgent validates the applicant's submitted documents against
the policy list and drafts the Turkish missing-document request template.
RAGDocumentAnalyzer loads applicant documents with LangChain loaders
(``TextLoader`` / ``PyMuPDFLoader``), chunks them and offers deterministic
lexical retrieval for the RAG step of the decision pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agents.base import AgentBase
from app.models import (
    DocumentCheckResult,
    DocumentRequirement,
    LoanApplication,
)
from app.models.rag import DocumentChunk, RAGAnalysis

_SUPPORTED_LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyMuPDFLoader,
}


class DocumentControlAgent(AgentBase):
    """Document validity + missing-document request drafting."""

    def __init__(self) -> None:
        super().__init__("document")

    def check(self, application: LoanApplication) -> DocumentCheckResult:
        self.logger.info(
            "Document check started for %s (%s)",
            application.applicant.name,
            application.applicant.identity_no,
        )
        requirements = [
            DocumentRequirement(code=item.code, description=item.description)
            for item in self.rules.document_policy.required_documents
        ]
        submitted = {code.upper() for code in application.applicant.submitted_documents}
        present = [req.code for req in requirements if req.code in submitted]
        missing = [req for req in requirements if req.code not in submitted]
        complete = not missing
        request_draft = self._build_request_draft(application, missing) if missing else ""
        self.logger.info(
            "Document check finished: complete=%s present=%d missing=%d",
            complete,
            len(present),
            len(missing),
        )
        return DocumentCheckResult(
            present=present, missing=missing, complete=complete, request_draft=request_draft
        )

    @staticmethod
    def _build_request_draft(
        application: LoanApplication, missing: list[DocumentRequirement]
    ) -> str:
        lines = [
            f"Sayın {application.applicant.name},",
            "",
            "Kredi başvurunuzun değerlendirmesine devam edebilmek için aşağıdaki "
            "belgeleri eksiksiz iletmeniz gerekmektedir:",
            "",
        ]
        lines.extend(f"- {req.code}: {req.description}" for req in missing)
        lines.extend(
            [
                "",
                "Lütfen belgelerinizi en geç 10 iş günü içinde şubemize veya "
                "müşteri hizmetlerimize iletiniz.",
                "Saygılarımızla,",
                "Kredi Operasyon Birimi",
            ]
        )
        return "\n".join(lines)


class RAGDocumentAnalyzer(AgentBase):
    """Loads, chunks and retrieves applicant documents via LangChain."""

    _loaders = _SUPPORTED_LOADERS

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        super().__init__("rag")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunks: list[DocumentChunk] = []

    def _splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def analyze(self, paths: Sequence[str]) -> RAGAnalysis:
        """Load every supported document, chunk it and index chunks in memory."""
        from app.core.config import get_settings

        settings = get_settings()
        chunk_size = self._chunk_size or settings.document_chunk_size
        chunk_overlap = self._chunk_overlap or settings.document_chunk_overlap
        splitter = self._splitter(chunk_size, chunk_overlap)

        loaded = []
        for path in paths:
            loader_type = self._loader_for(path)
            if loader_type is None:
                self.logger.warning("Skipping unsupported document: %s", path)
                continue
            if loader_type is TextLoader:
                loader = loader_type(path, encoding="utf-8")
            else:
                loader = loader_type(path)
            loaded.extend(loader.load())

        chunks: list[DocumentChunk] = []
        sources: set[str] = set()
        for document in loaded:
            source = document.metadata.get("source", "<unknown>")
            page = document.metadata.get("page")
            sources.add(source)
            for piece in splitter.split_text(document.page_content):
                chunks.append(
                    DocumentChunk(
                        source=source,
                        text=piece,
                        page=int(page) if isinstance(page, int) else None,
                    )
                )

        self._chunks = chunks
        self.logger.info("RAG analysis complete: %d sources, %d chunks", len(sources), len(chunks))
        return RAGAnalysis(chunks=chunks, sources=sorted(sources))

    def retrieve(self, query: str, k: int = 3) -> list[DocumentChunk]:
        """Top-k deterministic lexical retrieval over the indexed chunks."""
        tokens = [token for token in query.lower().split() if token]
        if not tokens or not self._chunks:
            return []

        def score(chunk: DocumentChunk) -> int:
            text_lower = chunk.text.lower()
            return sum(text_lower.count(token) for token in tokens)

        ranked = sorted(self._chunks, key=score, reverse=True)
        hits = [chunk for chunk in ranked if score(chunk) > 0][:k]
        return hits

    @classmethod
    def _loader_for(cls, path: str):
        from pathlib import Path

        return cls._loaders.get(Path(path).suffix.lower())
