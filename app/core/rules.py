"""Typed loaders for the versioned YAML rule files under ``rules/``.

No business threshold lives in code: policy rules and PD cut-offs, pricing
parameters and taxes, the authority matrix, workflow SLAs and the reason-code
dictionary are all read from here (and versioned by their ``version`` key).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings


class ProductPolicy(BaseModel):
    label: str
    max_dsr: float
    min_amount: float
    max_amount: float
    min_term: int
    max_term: int
    income_multiplier: float


class DecisionThresholds(BaseModel):
    auto_approve_max_pd: float
    auto_decline_min_pd: float
    min_document_confidence: float
    counter_offer_min_ratio: float = 0.3


class RiskBand(BaseModel):
    band: str
    max_pd: float
    limit_factor: float


class PolicyRule(BaseModel):
    id: str
    description: str
    when: str
    action: Literal["decline", "refer"]
    reason: str


class PolicyRules(BaseModel):
    version: str
    description: str = ""
    effective_from: str = ""
    products: dict[str, ProductPolicy]
    decision: DecisionThresholds
    risk_bands: list[RiskBand]
    rules: list[PolicyRule]

    def product(self, code: str) -> ProductPolicy:
        return self.products.get(code) or self.products["IHTIYAC"]

    def band_for(self, pd: float) -> RiskBand:
        for band in self.risk_bands:
            if pd <= band.max_pd:
                return band
        return self.risk_bands[-1]


class ProductPricing(BaseModel):
    lgd: float
    funding_cost_annual: float
    opex_annual: float
    target_raroc: float
    upfront_fee_rate: float
    min_rate_annual: float


class PricingConfig(BaseModel):
    version: str
    taxes: dict[str, float]
    legal_cap_annual: float
    offer_validity_days: int
    asset_correlation: float = 0.15
    confidence_level: float = 0.999
    products: dict[str, ProductPricing]

    def product(self, code: str) -> ProductPricing:
        return self.products.get(code) or self.products["IHTIYAC"]

    @property
    def tax_multiplier(self) -> float:
        return 1.0 + self.taxes.get("bsmv", 0.0) + self.taxes.get("kkdf", 0.0)


class AuthorityLevel(BaseModel):
    role: str
    max_amount: float
    max_pd: float


class FourEyes(BaseModel):
    amount_threshold: float
    pd_threshold: float
    overrides: bool = True


class AuthorityMatrix(BaseModel):
    version: str
    levels: list[AuthorityLevel]
    four_eyes: FourEyes


class WorkflowConfig(BaseModel):
    version: str
    sla_hours: dict[str, int]
    objection_window_days: int
    velocity_window_days: int
    fraud_ring_min_size: int
    data_collection_max_retries: int
    data_collection_retry_seconds: int
    queue_priority: dict[str, float]
    income_tolerance: float = 0.15


class ReasonCatalog(BaseModel):
    version: str
    codes: dict[str, str]
    feature_reasons: dict[str, str] = Field(default_factory=dict)


class SanctionEntry(BaseModel):
    name: str
    type: str
    list: str


class SanctionsList(BaseModel):
    version: str
    entries: list[SanctionEntry]


def _rules_dir() -> Path:
    return get_settings().rules_path


@lru_cache(maxsize=32)
def _read(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def read_yaml(name: str) -> dict[str, Any]:
    return _read(str(_rules_dir() / name))


def parse_policy(text: str) -> PolicyRules:
    return PolicyRules.model_validate(yaml.safe_load(text))


@lru_cache(maxsize=1)
def load_policy_file() -> PolicyRules:
    return PolicyRules.model_validate(read_yaml("policy_v1.yaml"))


def policy_file_text() -> str:
    return (_rules_dir() / "policy_v1.yaml").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_pricing() -> PricingConfig:
    return PricingConfig.model_validate(read_yaml("pricing.yaml"))


@lru_cache(maxsize=1)
def load_authority() -> AuthorityMatrix:
    return AuthorityMatrix.model_validate(read_yaml("authority_matrix.yaml"))


@lru_cache(maxsize=1)
def load_workflow() -> WorkflowConfig:
    return WorkflowConfig.model_validate(read_yaml("workflow.yaml"))


@lru_cache(maxsize=1)
def load_reasons() -> ReasonCatalog:
    return ReasonCatalog.model_validate(read_yaml("reason_codes.yaml"))


@lru_cache(maxsize=1)
def load_sanctions() -> SanctionsList:
    return SanctionsList.model_validate(read_yaml("reference/sanctions.yaml"))
