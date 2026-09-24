"""Read-only tools of the underwriter agent.

Every tool reads persisted data of *one* application (bound at construction)
and returns JSON-serialisable observations. Numbers the memo may quote are
exposed as citable facts with ``field_id`` keys so the citation guard can
verify them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.narrator import Fact
from app.agents.rag import policy_kb
from app.db.models import Application, BureauReport, CashflowFeatures, Document, ExtractedField
from app.workflow.narratives import decision_facts
from app.workflow.pipeline import latest_decision, latest_offer

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_application",
        "description": "Başvuru özeti: ürün, tutar, vade, beyan edilen gelir, çalışma şekli.",
        "parameters": {},
    },
    {
        "name": "get_documents_fields",
        "description": "Belgelerden çıkarılan alanlar, güven skorları ve sahtecilik sinyalleri.",
        "parameters": {},
    },
    {
        "name": "get_bureau_report",
        "description": "KKB/Findeks raporu ve SGK hizmet kaydı.",
        "parameters": {},
    },
    {
        "name": "get_cashflow_features",
        "description": "Açık bankacılık nakit akışı özellikleri.",
        "parameters": {},
    },
    {
        "name": "run_decision",
        "description": "Karar motoru sonucu: sonuç, PD, bant, skor kartı puanı, limitler.",
        "parameters": {},
    },
    {
        "name": "get_reason_codes",
        "description": "Başvurana özel gerekçe kodları ve karşı-olgusal öneriler.",
        "parameters": {},
    },
    {
        "name": "get_pricing",
        "description": "Risk bazlı fiyatlama: faiz, taksit, yıllık maliyet oranı, RAROC.",
        "parameters": {},
    },
    {
        "name": "search_policy",
        "description": "İç kredi politikası ve BDDK/KVKK özetlerinde arama.",
        "parameters": {"query": {"type": "string"}},
    },
]


def openai_tools() -> list[dict[str, Any]]:
    out = []
    for spec in TOOL_SPECS:
        properties = spec["parameters"]
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                    },
                },
            }
        )
    out.append(
        {
            "type": "function",
            "function": {
                "name": "draft_memo",
                "description": "Yeterli bilgi toplandığında kredi tahsis memorandumunu yazmaya geç.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    return out


class UnderwriterTools:
    def __init__(self, session: Session, application: Application) -> None:
        self.session = session
        self.app = application
        self.facts: dict[str, Fact] = {}

    def _add(self, *facts: Fact) -> None:
        for fact in facts:
            self.facts[fact.id] = fact

    @property
    def registry(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "get_application": self.get_application,
            "get_documents_fields": self.get_documents_fields,
            "get_bureau_report": self.get_bureau_report,
            "get_cashflow_features": self.get_cashflow_features,
            "run_decision": self.run_decision,
            "get_reason_codes": self.get_reason_codes,
            "get_pricing": self.get_pricing,
            "search_policy": self.search_policy,
        }

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        fn = self.registry.get(name)
        if fn is None:
            return {"error": f"bilinmeyen araç: {name}"}
        return fn(**(arguments or {})) if name == "search_policy" else fn()

    # ------------------------------------------------------------ tools
    def get_application(self) -> dict[str, Any]:
        app = self.app
        self._add(
            Fact(
                id="f:loan.amount",
                label="Talep edilen tutar",
                value=app.requested_amount,
                unit="TL",
            ),
            Fact(
                id="f:loan.term",
                label="Talep edilen vade",
                value=app.requested_term_months,
                unit="ay",
            ),
            Fact(
                id="f:income.declared",
                label="Beyan edilen gelir",
                value=app.declared_income,
                unit="TL",
            ),
        )
        return {
            "application_id": app.id,
            "product": app.product,
            "requested_amount": app.requested_amount,
            "requested_term_months": app.requested_term_months,
            "declared_income": app.declared_income,
            "employment_type": app.employment_type,
            "state": app.state,
        }

    def get_documents_fields(self) -> dict[str, Any]:
        docs = (
            self.session.execute(select(Document).where(Document.application_id == self.app.id))
            .scalars()
            .all()
        )
        fields = (
            self.session.execute(
                select(ExtractedField).where(ExtractedField.application_id == self.app.id)
            )
            .scalars()
            .all()
        )
        max_fraud = max((d.fraud_score for d in docs), default=0.0)
        min_conf = min((f.confidence for f in fields), default=1.0)
        self._add(
            Fact(
                id="f:docs.max_fraud_score",
                label="En yüksek belge sahtecilik skoru",
                value=round(max_fraud, 3),
            ),
            Fact(
                id="f:docs.min_confidence", label="En düşük alan güveni", value=round(min_conf, 3)
            ),
            Fact(id="f:docs.count", label="Belge sayısı", value=len(docs), unit="adet"),
        )
        recon = (self.app.kyc or {}).get("reconciliation", {})
        for label, check in (recon.get("checks") or {}).items():
            self._add(
                Fact(
                    id=f"f:recon.{label}",
                    label=f"Mutabakat ({label})",
                    value=check["value"],
                    unit="TL",
                )
            )
        return {
            "documents": [
                {
                    "code": d.code,
                    "status": d.status,
                    "fraud_score": d.fraud_score,
                    "signals": [s["label"] for s in d.fraud_signals],
                }
                for d in docs
            ],
            "fields": [
                {
                    "document": f.document_id[:8],
                    "name": f.name,
                    "value": f.value,
                    "confidence": f.confidence,
                }
                for f in fields
                if f.name != "tckn"
            ],
            "reconciliation": recon,
        }

    def get_bureau_report(self) -> dict[str, Any]:
        reports = {
            r.provider: r.payload
            for r in self.session.execute(
                select(BureauReport).where(BureauReport.application_id == self.app.id)
            ).scalars()
        }
        kkb = reports.get("KKB", {})
        sgk = reports.get("SGK", {})
        if kkb.get("score") is not None:
            self._add(
                Fact(
                    id="f:bureau.score",
                    label="KKB kredi notu",
                    value=int(kkb["score"]),
                    unit="puan",
                )
            )
        self._add(
            Fact(
                id="f:bureau.active_loans",
                label="Aktif kredi",
                value=int(kkb.get("active_loans") or 0),
                unit="adet",
            ),
            Fact(
                id="f:bureau.instalments",
                label="Mevcut aylık taksitler",
                value=float(kkb.get("monthly_instalments") or 0.0),
                unit="TL",
            ),
            Fact(
                id="f:bureau.delinquencies",
                label="Son 24 ay gecikme",
                value=int(kkb.get("delinquency_count_24m") or 0),
                unit="adet",
            ),
            Fact(
                id="f:employment.months",
                label="Çalışma süresi",
                value=int(sgk.get("employment_months") or 0),
                unit="ay",
            ),
        )
        return {
            "kkb": kkb,
            "sgk": {k: v for k, v in sgk.items() if k != "reported_gross_earnings_12m"},
        }

    def get_cashflow_features(self) -> dict[str, Any]:
        row = (
            self.session.execute(
                select(CashflowFeatures)
                .where(CashflowFeatures.application_id == self.app.id)
                .order_by(CashflowFeatures.id.desc())
            )
            .scalars()
            .first()
        )
        if row is None:
            return {"available": False}
        f = row.features
        self._add(
            Fact(
                id="f:cashflow.avg_income",
                label="Ortalama aylık maaş girişi",
                value=f.get("avg_monthly_income", 0.0),
                unit="TL",
            ),
            Fact(
                id="f:cashflow.income_cv", label="Gelir değişkenliği", value=f.get("income_cv", 0.0)
            ),
            Fact(
                id="f:cashflow.negative_days",
                label="Eksi bakiye gün",
                value=f.get("negative_balance_days", 0.0),
                unit="gün",
            ),
            Fact(
                id="f:cashflow.savings_rate",
                label="Tasarruf oranı",
                value=f.get("savings_rate", 0.0),
                unit="%",
            ),
            Fact(
                id="f:cashflow.gambling_share",
                label="Kumar/bahis payı",
                value=f.get("gambling_share", 0.0),
                unit="%",
            ),
        )
        return {"available": True, "features": f}

    def run_decision(self) -> dict[str, Any]:
        decision = latest_decision(self.session, self.app.id)
        if decision is None:
            return {"available": False}
        self._add(*decision_facts(self.app, decision))
        return {
            "outcome": decision.outcome,
            "conditional": decision.conditional,
            "pd": decision.pd,
            "risk_band": decision.risk_band,
            "score_points": decision.score_points,
            "limits": decision.limits,
            "fired_rules": [r["id"] for r in decision.rule_results if r.get("fired")],
            "challenger_pd": (decision.challenger or {}).get("pd"),
        }

    def get_reason_codes(self) -> dict[str, Any]:
        decision = latest_decision(self.session, self.app.id)
        if decision is None:
            return {"reason_codes": [], "counterfactuals": []}
        return {
            "reason_codes": decision.reason_codes,
            "counterfactuals": [c["text"] for c in decision.counterfactuals],
        }

    def get_pricing(self) -> dict[str, Any]:
        decision = latest_decision(self.session, self.app.id)
        offer = latest_offer(self.session, self.app.id)
        pricing = {
            k: v
            for k, v in ((decision.pricing if decision else {}) or {}).items()
            if k != "schedule"
        }
        if pricing:
            self._add(
                Fact(
                    id="f:pricing.raroc",
                    label="RAROC",
                    value=float(pricing.get("raroc", 0.0)),
                    unit="%",
                ),
                Fact(
                    id="f:pricing.expected_loss",
                    label="Beklenen kayıp",
                    value=float(pricing.get("expected_loss_annual", 0.0)),
                    unit="TL",
                ),
            )
        return {"pricing": pricing, "offer_status": offer.status if offer else None}

    def search_policy(self, query: str = "borç servis oranı") -> dict[str, Any]:
        hits = policy_kb().search(query, k=3)
        return {
            "query": query,
            "results": [
                {"chunk_id": h["chunk_id"], "title": h["title"], "text": h["text"][:600]}
                for h in hits
            ],
        }
