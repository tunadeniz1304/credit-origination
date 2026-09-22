"""Belge Kontrol Ajanı — validates the applicant's submitted documents.

Compares submitted document codes against the policy list in
``config/document_policy.required_documents``, flags anything missing, and
drafts a Turkish request template listing each missing document with its
Turkish description.
"""
from __future__ import annotations

from src.agents.base import AgentBase
from src.models import DocumentCheckResult, DocumentRequirement, LoanApplication


class DocumentControlAgent(AgentBase):
    """Document validity + missing-document request drafting."""

    def __init__(self) -> None:
        super().__init__("document")

    def check(self, application: LoanApplication) -> DocumentCheckResult:
        self.logger.info("Document check started for %s (%s)", application.applicant.name, application.applicant.identity_no)
        requirements = [
            DocumentRequirement(code=item["code"], description=item["description"])
            for item in self.config["document_policy"]["required_documents"]
        ]
        submitted = {code.upper() for code in application.applicant.submitted_documents}
        present = [req.code for req in requirements if req.code in submitted]
        missing = [req for req in requirements if req.code not in submitted]
        complete = not missing
        request_draft = self._build_request_draft(application, missing) if missing else ""
        self.logger.info(
            "Document check finished: complete=%s present=%d missing=%d",
            complete, len(present), len(missing),
        )
        return DocumentCheckResult(present=present, missing=missing, complete=complete, request_draft=request_draft)

    @staticmethod
    def _build_request_draft(application: LoanApplication, missing: list[DocumentRequirement]) -> str:
        lines = [
            f"Sayın {application.applicant.name},",
            "",
            "Kredi başvurunuzun değerlendirmesine devam edebilmek için aşağıdaki belgeleri eksiksiz iletmeniz gerekmektedir:",
            "",
        ]
        lines.extend(f"- {req.code}: {req.description}" for req in missing)
        lines.extend(
            [
                "",
                "Lütfen belgelerinizi en geç 10 iş günü içinde şubemize veya müşteri hizmetlerimize iletiniz.",
                "Saygılarımızla,",
                "Kredi Operasyon Birimi",
            ]
        )
        return "\n".join(lines)
