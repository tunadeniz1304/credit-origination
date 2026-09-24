"""ApplicationPipeline: orchestrates the async credit decision pipeline."""

from __future__ import annotations

from pathlib import Path

from app.agents.api_agent import ApiIntegrationAgent
from app.agents.committee_agent import CreditCommitteeAgent
from app.agents.document_agent import DocumentControlAgent, RAGDocumentAnalyzer
from app.agents.llm import LLMProvider
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.engine.reports import generate_report_files
from app.models import LoanApplication, PipelineResult


class ApplicationPipeline:
    """Runs one application through document control, data collection,
    committee decision and report generation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = get_logger("engine.pipeline")

    async def run(
        self,
        application: LoanApplication,
        llm: LLMProvider | None = None,
        application_id: str | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline for a single application."""
        self.logger.info(
            "Pipeline started for %s (%s)",
            application.applicant.name,
            application.applicant.identity_no,
        )
        document_check = DocumentControlAgent().check(application)
        rag_summary = self._analyze_documents(application)

        financial = await ApiIntegrationAgent(self.settings).collect(application.applicant)

        decision = CreditCommitteeAgent(self.settings).decide(
            application=application,
            financial=financial,
            llm=llm,
        )
        payload = PipelineResult(
            application=application,
            status=decision.status,
            document_check=document_check,
            decision=decision,
            rag_summary=rag_summary,
        )
        self.logger.info(
            "Pipeline finished for %s -> %s",
            application.applicant.identity_no,
            decision.status.value,
        )
        if application_id:
            generate_report_files(payload, application_id, self.settings)
            json_path, pdf_path = self._report_paths(application_id)
            payload = payload.model_copy(
                update={
                    "report_json_path": json_path,
                    "report_pdf_path": pdf_path,
                }
            )
        return payload

    def _report_paths(self, application_id: str) -> tuple[str, str]:
        """Resolve expected report file paths for an application id."""
        report_dir = self.settings.report_dir
        return str(report_dir / f"{application_id}_report.json"), str(
            report_dir / f"{application_id}_report.pdf"
        )

    def _analyze_documents(self, application: LoanApplication) -> str:
        """Run RAG over uploaded documents matching submitted codes, if any."""
        uploads = self.settings.uploads_dir
        if not uploads.is_dir():
            return ""
        submitted = {code.upper() for code in application.applicant.submitted_documents}
        paths: list[str] = []
        for file in sorted(uploads.iterdir()):
            if not file.is_file():
                continue
            if Path(file.name).stem.upper() in submitted:
                paths.append(str(file))
        if not paths:
            return ""
        analysis = RAGDocumentAnalyzer().analyze(paths)
        return f"{len(analysis.chunks)} chunks from {analysis.sources}"
