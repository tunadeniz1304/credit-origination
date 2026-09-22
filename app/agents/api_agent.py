"""API Entegrasyon Ajanı — aggregates applicant financial data.

Calls the resilient KKB (credit bureau) and e-Devlet (employment) clients
and assembles a single :class:`AggregatedFinancialData` picture for the
decision engine.
"""
from __future__ import annotations

import asyncio

from app.agents.base import AgentBase
from app.core.config import Settings, get_settings
from app.integrations.clients import EDevletClient, KKBClient
from app.models import AggregatedFinancialData, Applicant


class ApiIntegrationAgent(AgentBase):
    """Aggregates applicant financial data via KKB / e-Devlet clients."""

    def __init__(
        self,
        settings: Settings | None = None,
        kkb: KKBClient | None = None,
        edevlet: EDevletClient | None = None,
    ) -> None:
        super().__init__("api")
        settings = settings or get_settings()
        self._kkb = kkb or KKBClient(settings)
        self._edevlet = edevlet or EDevletClient(settings)

    async def collect(self, applicant: Applicant) -> AggregatedFinancialData:
        identity_no = applicant.identity_no
        self.logger.info("External data collection started for %s", identity_no)
        credit, employment = await asyncio.gather(
            self._kkb.get_report(identity_no),
            self._edevlet.get_employment(identity_no),
        )
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
            data.kbb_score,
            data.risk_class,
            data.total_debt,
            data.employer,
            data.employment_years,
        )
        return data
