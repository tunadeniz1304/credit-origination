"""Transparent committee factor table + narrated rationale.

The committee factors (bureau score, debt service ratio, loan-to-income and
term) are the human-readable threshold view of an application. The verdict is
deterministic; the rationale is produced by :class:`~app.agents.narrator.Narrator`
(live LLM chain or the deterministic Turkish templates) and never influences
the outcome.
"""

from __future__ import annotations

from app.agents.base import AgentBase
from app.agents.llm_service import LLMService
from app.agents.narrator import Fact, NarrativeContext, Narrator, ReasonText
from app.core.config import Settings, get_settings
from app.engine.schedule import build_schedule, monthly_instalment
from app.models import (
    AggregatedFinancialData,
    ApplicationStatus,
    CommitteeDecision,
    CommitteeFactor,
    LoanApplication,
)

FACTOR_KBB = "Kredi Skoru (KKB)"
FACTOR_DSR = "Borç Servis Oranı"
FACTOR_LTI = "Kredi/Gelir Çarpanı"
FACTOR_TERM = "Vade"
# Existing bureau balances are assumed to amortise over this many months when
# the bureau does not report instalments explicitly.
EXISTING_DEBT_AMORTISATION_MONTHS = 36


def debt_service_ratio(
    monthly_income: float,
    existing_monthly_debt_service: float,
    new_instalment: float,
) -> float:
    """DSR = (existing monthly instalments + new instalment) / monthly net income."""
    if monthly_income <= 0:
        return float("inf")
    return (existing_monthly_debt_service + new_instalment) / monthly_income


class CreditCommitteeAgent(AgentBase):
    """Deterministic threshold gates + narrated (never deciding) rationale."""

    def __init__(self, settings: Settings | None = None, llm: LLMService | None = None) -> None:
        super().__init__("committee")
        self._settings = settings or get_settings()
        self._llm = llm

    def evaluate(
        self, application: LoanApplication, financial: AggregatedFinancialData
    ) -> tuple[list[CommitteeFactor], float, float]:
        """Return (factors, dsr, max approvable amount)."""
        committee = self.rules.committee
        income = financial.monthly_income
        term = min(application.requested_term_months, committee.max_term_months)
        existing = monthly_instalment(financial.total_debt, EXISTING_DEBT_AMORTISATION_MONTHS)
        new_instalment = monthly_instalment(application.requested_amount, term)
        dsr = debt_service_ratio(income, existing, new_instalment)
        lti = application.requested_amount / income
        capacity = max(0.0, committee.max_debt_service_ratio * income - existing)
        per_unit = monthly_instalment(1.0, term)
        max_amount = min(capacity / per_unit, income * committee.max_loan_to_income_multiplier)
        factors = [
            CommitteeFactor(
                name=FACTOR_KBB,
                value=float(financial.kbb_score),
                threshold=float(committee.min_kbb_score),
                operator=">=",
                passed=financial.kbb_score >= committee.min_kbb_score,
                unit=" puan",
            ),
            CommitteeFactor(
                name=FACTOR_DSR,
                value=round(dsr, 4),
                threshold=committee.max_debt_service_ratio,
                operator="<=",
                passed=dsr <= committee.max_debt_service_ratio,
            ),
            CommitteeFactor(
                name=FACTOR_LTI,
                value=round(lti, 4),
                threshold=committee.max_loan_to_income_multiplier,
                operator="<=",
                passed=lti <= committee.max_loan_to_income_multiplier,
                unit="x",
            ),
            CommitteeFactor(
                name=FACTOR_TERM,
                value=float(application.requested_term_months),
                threshold=float(committee.max_term_months),
                operator="<=",
                passed=application.requested_term_months <= committee.max_term_months,
                unit=" ay",
            ),
        ]
        return factors, dsr, round(max_amount, 2)

    async def decide(
        self,
        application: LoanApplication,
        financial: AggregatedFinancialData,
        application_id: str = "",
    ) -> CommitteeDecision:
        factors, dsr, max_amount = self.evaluate(application, financial)
        approved = all(f.passed for f in factors)
        status = ApplicationStatus.APPROVED if approved else ApplicationStatus.REJECTED
        term = min(application.requested_term_months, self.rules.committee.max_term_months)
        # Approved: never more than requested. Rejected: the capacity figure is
        # only a counterfactual hint, never an approved amount.
        suggested = min(application.requested_amount, max_amount) if approved else 0.0
        counterfactual = round(max(0.0, max_amount), -3) if not approved else None

        ctx = self._context(
            application,
            financial,
            application_id,
            status,
            dsr,
            factors,
            suggested,
            term,
            counterfactual,
        )
        bundle = await Narrator(self._llm or LLMService(self._settings)).run_chain(ctx)
        self.logger.info(
            "Committee decision for %s: %s (llm=%s)",
            application_id,
            status.value,
            bundle.overall_mode,
        )
        return CommitteeDecision(
            status=status,
            approved=approved,
            factors=factors,
            rationale=bundle.committee_summary,
            applicant_letter=bundle.applicant_letter,
            llm_mode=bundle.overall_mode,
            llm_error_kind=next(iter(bundle.errors.values()), None),
            debt_service_ratio=round(dsr, 4),
            suggested_amount=suggested,
            suggested_term_months=term,
            counterfactual_amount=counterfactual,
        )

    def _context(
        self,
        application: LoanApplication,
        financial: AggregatedFinancialData,
        application_id: str,
        status: ApplicationStatus,
        dsr: float,
        factors: list[CommitteeFactor],
        suggested: float,
        term: int,
        counterfactual: float | None,
    ) -> NarrativeContext:
        committee = self.rules.committee
        facts = [
            Fact(
                id="f:income.monthly",
                label="Aylık net gelir",
                value=financial.monthly_income,
                unit="TL",
            ),
            Fact(
                id="f:loan.amount",
                label="Talep edilen tutar",
                value=application.requested_amount,
                unit="TL",
            ),
            Fact(
                id="f:loan.term", label="Vade", value=application.requested_term_months, unit="ay"
            ),
            Fact(id="f:dsr", label="Borç servis oranı", value=round(dsr, 4), unit="%"),
            Fact(
                id="f:policy.max_dsr",
                label="DSR sınırı",
                value=committee.max_debt_service_ratio,
                unit="%",
            ),
            Fact(
                id="f:bureau.score", label="KKB kredi notu", value=financial.kbb_score, unit="puan"
            ),
            Fact(
                id="f:employment.months",
                label="Çalışma süresi",
                value=financial.employment_years * 12,
                unit="ay",
            ),
        ]
        if status == ApplicationStatus.APPROVED:
            schedule = build_schedule(application_id, principal=suggested, term_months=term)
            facts += [
                Fact(id="f:offer.amount", label="Teklif tutarı", value=suggested, unit="TL"),
                Fact(id="f:offer.term", label="Teklif vadesi", value=term, unit="ay"),
                Fact(
                    id="f:offer.instalment",
                    label="Aylık taksit",
                    value=schedule.instalment,
                    unit="TL",
                ),
                Fact(
                    id="f:pricing.annual_rate",
                    label="Yıllık faiz",
                    value=schedule.annual_rate,
                    unit="%",
                ),
            ]
        reasons = [
            ReasonText(code=f"K{i + 1:02d}", text=f"{f.name} eşiği karşılanmadı: {f.summary}.")
            for i, f in enumerate(factors)
            if not f.passed
        ]
        counterfactuals = []
        if counterfactual:
            facts.append(
                Fact(id="f:limit.max_amount", label="Azami tutar", value=counterfactual, unit="TL")
            )
            counterfactuals.append(
                f"Talep tutarını {Fact(id='x', label='', value=counterfactual, unit='TL').display} "
                "veya altına düşürmeniz borç servis oranınızı politika sınırına çeker."
            )
        return NarrativeContext(
            application_id=application_id or "BASVURU",
            applicant_name=application.applicant.name,
            outcome="OTOMATIK_ONAY" if status == ApplicationStatus.APPROVED else "OTOMATIK_RET",
            facts=facts,
            reason_codes=reasons,
            counterfactuals=counterfactuals,
            redactor_fields={"identity_no": application.applicant.identity_no},
        )
