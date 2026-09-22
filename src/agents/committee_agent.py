"""Kredi Komitesi Ajanı — autonomous approve/reject decision with rationale.

Evaluates the aggregated financial data against the committee thresholds in
``config/committee`` and prepares a structured Turkish report for the human
credit committee: per-factor pass/fail lines, an overall verdict, and a
suggested amount/term.
"""
from __future__ import annotations

from src.agents.base import AgentBase
from src.models import (
    AggregatedFinancialData,
    ApplicationStatus,
    CommitteeDecision,
    CommitteeFactor,
    LoanApplication,
)


class CreditCommitteeAgent(AgentBase):
    """Applies committee policy thresholds and drafts the decision report."""

    def __init__(self) -> None:
        super().__init__("committee")

    def decide(
        self, application: LoanApplication, financial: AggregatedFinancialData
    ) -> CommitteeDecision:
        self.logger.info("Committee review started for %s (%s)", application.applicant.name, application.applicant.identity_no)
        thresholds = self.config["committee"]
        annual_income = financial.monthly_income * 12

        factors = [
            CommitteeFactor(
                name="KKB (kredi notu) skoru",
                value=float(financial.kbb_score),
                threshold=float(thresholds["min_kbb_score"]),
                operator=">=",
                passed=financial.kbb_score >= thresholds["min_kbb_score"],
                unit=" puan",
            ),
            CommitteeFactor(
                name="Toplam borç / yıllık gelir oranı",
                value=financial.total_debt / annual_income,
                threshold=float(thresholds["max_debt_to_income_ratio"]),
                operator="<=",
                passed=(financial.total_debt / annual_income) <= thresholds["max_debt_to_income_ratio"],
            ),
            CommitteeFactor(
                name="Talep tutarı / yıllık gelir çarpanı",
                value=application.requested_amount / annual_income,
                threshold=float(thresholds["max_loan_to_income_multiplier"]),
                operator="<=",
                passed=(application.requested_amount / annual_income) <= thresholds["max_loan_to_income_multiplier"],
                unit="x",
            ),
            CommitteeFactor(
                name="Vade süresi",
                value=float(application.requested_term_months),
                threshold=float(thresholds["max_term_months"]),
                operator="<=",
                passed=application.requested_term_months <= thresholds["max_term_months"],
                unit=" ay",
            ),
        ]

        approved = all(factor.passed for factor in factors)
        status = ApplicationStatus.APPROVED if approved else ApplicationStatus.REJECTED
        decision = CommitteeDecision(
            status=status,
            approved=approved,
            factors=factors,
            rationale=self._build_report(application, financial, factors, approved),
            suggested_amount=application.requested_amount if approved else 0.0,
            suggested_term_months=application.requested_term_months if approved else 0,
        )
        self.logger.info(
            "Committee review finished: %s (passed=%d/%d)",
            status.value, sum(f.passed for f in factors), len(factors),
        )
        return decision

    @staticmethod
    def _build_report(
        application: LoanApplication,
        financial: AggregatedFinancialData,
        factors: list[CommitteeFactor],
        approved: bool,
    ) -> str:
        employment = (
            financial.employer
            + f" ({financial.employment_years} yıl) — "
            + ("doğrulandı" if financial.employment_verified else "doğrulama bekliyor")
        )
        lines = [
            "KREDİ KOMİTESİ DEĞERLENDİRME RAPORU",
            "===================================",
            f"Başvuru Sahibi : {application.applicant.name}  (TCKN: {application.applicant.identity_no})",
            f"Talep Edilen   : {application.requested_amount:,.0f} {application.currency} / {application.requested_term_months} ay",
            f"Aylık Gelir    : {financial.monthly_income:,.0f} {application.currency}",
            f"KKB Skoru      : {financial.kbb_score} ({financial.risk_class})",
            f"Toplam Borç    : {financial.total_debt:,.0f} {application.currency}",
            f"İstihdam Kaydı : {employment}",
            "",
            "KARAR FAKTÖRLERİ",
            "-----------------",
        ]
        lines.extend(f"[{'PASS' if f.passed else 'FAIL'}] {f.summary}" for f in factors)
        lines.extend(["", "KOMİTE KARARI", "-------------"])
        if approved:
            lines.extend(
                [
                    "ONAYLANDI: Başvuru, komite eşiklerinin tamamını karşılamaktadır.",
                    f"Önerilen Tahsis : {application.requested_amount:,.0f} {application.currency}, "
                    f"{application.requested_term_months} ay vade.",
                ]
            )
        else:
            failed_names = [f.name for f in factors if not f.passed]
            lines.extend(
                [
                    "REDDEDİLDİ: Aşağıdaki kriter(ler) komite eşiklerini karşılamamaktadır:",
                    *[f"- {name}" for name in failed_names],
                    "Başvuru sahibinin başvurusu bu gerekçelerle onaylanmamıştır.",
                ]
            )
        return "\n".join(lines)
