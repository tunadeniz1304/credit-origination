"""RAG document analysis + Turkish document-control contract tests."""
from __future__ import annotations

from pathlib import Path

from app.agents.document_agent import DocumentControlAgent, RAGDocumentAnalyzer
from app.models import Applicant, LoanApplication

_ALL_DOCS = ["IDENTITY", "INCOME", "EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"]


def _applicant(name: str, docs: list[str]) -> Applicant:
    return Applicant(
        name=name,
        identity_no="12345678901",
        monthly_income=30_000.0,
        submitted_documents=docs,
    )


def _text_file(tmp_path: Path, content: str, name: str = "gelir_belgesi.txt") -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_rag_analyze_chunks_text_document(tmp_path):
    content = "\n\n".join(
        f"Paragraf {i}: Aylık gelir 30000 TRY, banka ekstresi dökümü ve ikamet bilgileri bu belgede yer alır."
        for i in range(30)
    )
    path = _text_file(tmp_path, content)
    analysis = RAGDocumentAnalyzer().analyze([path])
    assert len(analysis.sources) == 1
    assert analysis.sources[0].endswith("gelir_belgesi.txt")
    assert analysis.total_chars > 0
    assert analysis.total_chunks >= 1
    assert all(chunk.source.endswith("gelir_belgesi.txt") for chunk in analysis.chunks)


def test_rag_retrieval_returns_matching_chunks(tmp_path):
    content = "\n\n".join(
        [
            "Maaş bordrosu: maaş ödemeleri, kesintiler ve primler listelenir.",
            "İkametgah belgesi: adres bilgileri ve ikamet süresi doğrulanır.",
            "Banka hesap ekstresi: son üç aydaki hesap hareketleri görüntülenir.",
        ]
        * 15
    )
    path = _text_file(tmp_path, content)
    analyzer = RAGDocumentAnalyzer(chunk_size=120, chunk_overlap=10)
    analyzer.analyze([path])
    hits = analyzer.retrieve("ekstresi", k=3)
    assert hits, "expected at least one retrieval hit"
    assert all("ekstresi" in hit.text.lower() for hit in hits)


def test_document_control_complete_when_all_present():
    application = LoanApplication(
        applicant=_applicant("Ali Yılmaz", list(_ALL_DOCS)),
        requested_amount=120_000.0,
        requested_term_months=36,
    )
    result = DocumentControlAgent().check(application)
    assert result.complete is True
    assert result.missing == []
    assert result.request_draft == ""


def test_document_control_drafts_turkish_request():
    application = LoanApplication(
        applicant=_applicant("Ayşe Demir", ["IDENTITY", "INCOME"]),
        requested_amount=120_000.0,
        requested_term_months=36,
    )
    result = DocumentControlAgent().check(application)
    assert result.complete is False
    missing_codes = {req.code for req in result.missing}
    assert missing_codes == {"EMPLOYMENT", "ADDRESS", "BANK_STATEMENT"}
    draft = result.request_draft
    assert "Ayşe Demir" in draft
    assert "Son 3 Ay Banka Hesap Ekstresi" in draft  # Turkish description surfaced
    assert "Kredi Operasyon Birimi" in draft  # template footer
