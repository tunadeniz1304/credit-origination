"""API Entegrasyon Ajanı — collects financial data from mock external sources.

Calls the mock KKB (credit bureau) and e-Devlet (employment) providers and
aggregates the results into a single :class:`AggregatedFinancialData` picture.
"""
from __future__ import annotations

from src.agents.base import AgentBase
from src.integrations.mock_providers import EDevletClient, KKBClient
from src.models import AggregatedFinancialData, Applicant


class ApiIntegrationAgent(AgentBase):
    """Aggregates applicant financial data via mock KKB / e-Devlet calls."""

    def __init__(self) -> None:
        super().__init__("api")
        self._kkb = KKBClient()
        self._edevlet = EDevletClient()

    @property
    def kkb_calls(self) -> int:
        return self._kkb.calls

    @property
    def edevlet_calls(self) -> int:
        return self._edevlet.calls

    def collect(self, applicant: Applicant) -> AggregatedFinancialData:
        identity_no = applicant.identity_no
        self.logger.info("External data collection started for %s", identity_no)
        credit = self._kkb.get_report(identity_no)
        employment = self._edevlet.get_employment(identity_no)
        data = AggregatedFinancialData(
            identity_no=identity_no,
            kbb_score=credit.score,
            risk_class=credit.risk_class,
            total_debt=credit.total_debt,
            monthly_income=applicant.monthly_income,
            employer=employment.employer,
            employment_years=employment.years,
            employment_verified=employment.verified,
        )
        self.logger.info(
            "External data collected: kbb=%d risk='%s' debt=%.2f employer='%s' years=%d",
            data.kbb_score, data.risk_class, data.total_debt, data.employer, data.employment_years,
        )
        return data
