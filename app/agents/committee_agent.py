"""Kredi Tahsis Komitesi: deterministic rule evaluation + Turkish rationale.

CreditCommitteeAgent evaluates the applicant's aggregated financial data
against the committee thresholds from ``config/config.json``. The verdict is
fully deterministic (all four factors must pass); the accompanying rationale
report is produced by a two-step Turkish prompt chain over the active LLM
provider, so the business decision never depends on model randomness while
the narrative stays explainable and BDDK-report-shaped.
"""
from __future__ import annotations

from app.agents.base import AgentBase
from app.agents.llm import LLMProvider, get_provider
from app.core.config import Settings, get_settings
from app.models import (
    AggregatedFinancialData,
    ApplicationStatus,
    CommitteeDecision,
    CommitteeFactor,
    LoanApplication,
)

_FACTOR_KBB = "Kredi Skoru (KKB)"
_FACTOR_DTI = "Borç/Gelir Oranı"
_FACTOR_LTI = "Kredi/Gelir Çarpanı"
_FACTOR_TERM = "Vade"


class CreditCommitteeAgent(AgentBase):
    """Deterministic credit committee: rule gates + prompt-chained rationale."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__("committee")
        self._settings = settings

    def decide(
        self,
        application: LoanApplication,
        financial: AggregatedFinancialData,
        llm: LLMProvider | None = None,
    ) -> CommitteeDecision:
        """Evaluate the four committee gates and build the Turkish report."""
        committee = self.rules.committee
        provider = llm or get_provider(self._settings or get_settings())

        factors = [
            CommitteeFactor(
                name=_FACTOR_KBB,
                value=float(financial.kbb_score),
                threshold=float(max(0, committee.min_kbb_score)),
                operator=">=",
                passed=financial.kbb_score >= committee.min_kbb_score,
                unit=" puan",
            ),
            CommitteeFactor(
                name=_FACTOR_DTI,
                value=financial.total_debt / financial.monthly_income,
                threshold=committee.max_debt_to_income_ratio,
                operator="<=",
                passed=financial.total_debt / financial.monthly_income
                <= committee.max_debt_to_income_ratio,
            ),
            CommitteeFactor(
                name=_FACTOR_LTI,
                value=application.requested_amount / financial.monthly_income,
                threshold=committee.max_loan_to_income_multiplier,
                operator="<=",
                passed=application.requested_amount / financial.monthly_income
                <= committee.max_loan_to_income_multiplier,
                unit="x",
            ),
            CommitteeFactor(
                name=_FACTOR_TERM,
                value=float(application.requested_term_months),
                threshold=float(committee.max_term_months),
                operator="<=",
                passed=application.requested_term_months <= committee.max_term_months,
                unit=" ay",
            ),
        ]

        all_passed = all(factor.passed for factor in factors)
        status = ApplicationStatus.APPROVED if all_passed else ApplicationStatus.REJECTED
        suggested_amount = financial.monthly_income * committee.max_loan_to_income_multiplier
        suggested_term = min(
            application.requested_term_months,
            committee.max_term_months,
        )

        rationale = self._build_rationale(
            provider=provider,
            application=application,
            financial=financial,
            factors=factors,
            status=status,
        )

        self.logger.info(
            "Committee decision for %s: %s (all_passed=%s)",
            application.applicant.identity_no,
            status.value,
            all_passed,
        )
        return CommitteeDecision(
            status=status,
            approved=all_passed,
            factors=factors,
            rationale=rationale,
            suggested_amount=suggested_amount,
            suggested_term_months=suggested_term,
        )

    def _build_rationale(
        self,
        provider: LLMProvider,
        application: LoanApplication,
        financial: AggregatedFinancialData,
        factors: list[CommitteeFactor],
        status: ApplicationStatus,
    ) -> str:
        """Two-step Turkish prompt chain; both steps overridable by the provider."""
        factor_lines = "\n".join(f"- {factor.summary}" for factor in factors)
        applicant = application.applicant

        step_one = provider.complete(
            system=(
                "Sen Türkiye'de faaliyet gösteren bir bankada Kredi Tahsis Komitesi "
                "raporlama uzmanısın. Raporu resmi, nesnel ve gerekçeli şekilde Türkçe yaz."
            ),
            user=(
                "Aşağıdaki kredi başvurusu ve KKB/e-Devlet verileri doğrultusunda "
                "derin ve gerekçeli bir değerlendirme raporu yaz.\n\n"
                f"Başvuru Sahibi: {applicant.name} (T.C. No: {applicant.identity_no})\n"
                f"Aylık Gelir: {financial.monthly_income:,.2f} TRY\n"
                f"Talep Edilen Kredi: {application.requested_amount:,.2f} {application.currency}\n"
                f"Vade: {application.requested_term_months} ay\n"
                f"KKB Skoru: {financial.kbb_score} ({financial.risk_class})\n"
                f"Toplam Borç: {financial.total_debt:,.2f} TRY\n"
                f"İşveren: {financial.employer} ({financial.employment_years} yıl, "
                f"doğrulanmış={financial.employment_verified})\n\n"
                "Komite Eşik Değerlendirmeleri:\n"
                f"{factor_lines}\n\n"
                "Rapor; başvuru sahibinin ödeme gücü, kaldıraç ve vade uygunluğunu tartışsın, "
                "her eşiği tek tek yorumlasın ve nihai öneriyi gerekçeleriyle desteklesin."
            ),
            temperature=0.0,
        ).strip()

        step_two = provider.complete(
            system=(
                "Sen Türkiye'de bir bankanın Kredi Tahsis Komitesi sekreterisin. "
                "Öneriyi kısa, resmi ve öz bir biçimde Türkçe yaz."
            ),
            user=(
                f"Komite kararı: {status.value}.\n"
                f"Başvuru Sahibi: {applicant.name} (T.C. No: {applicant.identity_no}).\n"
                "Komite önerisini iki ila üç cümleyle özetle ve gerekçeyi belirt."
            ),
            temperature=0.0,
        ).strip()

        if not step_one and not step_two:
            # Absolute fallback so rationale is never empty even with a stubbed LLM.
            step_two = (
                f"Komite değerlendirmesi tamamlandı. "
                f"{applicant.name} başvurusu için nihai karar: {status.value}."
            )
        return f"{step_one}\n\nKurul Kararı: {step_two}" if step_one else step_two
