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


class RetailCorrelation(BaseModel):
    """Basel IRB 'other retail' asset correlation parameters."""

    r_min: float = 0.03
    r_max: float = 0.16
    k: float = 35.0


class PricingConfig(BaseModel):
    version: str
    taxes: dict[str, float]
    legal_cap_annual: float
    offer_validity_days: int
    correlation: RetailCorrelation = Field(default_factory=RetailCorrelation)
    confidence_level: float = 0.999
    capital_floor: float = 0.01
    pd_floor: float = 0.0003
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


class ChampionSelection(BaseModel):
    significance_level: float
    min_auc_gain: float
    simplicity_order: list[str]


class FairnessConfig(BaseModel):
    air_threshold: float
    approval_rate: float
    age_bins: list[float]
    age_labels: list[str]
    min_group_size: int
    min_group_share: float = 0.0

    def group_floor(self, n: int) -> int:
        return max(self.min_group_size, round(self.min_group_share * n))


class LDAConfig(BaseModel):
    attribute: str
    proxy_auc_threshold: float
    eg_epsilons: list[float]
    max_auc_loss: float


class LaneBConfig(BaseModel):
    band_tolerance_abs: float
    low_risk_tolerance_ratio: float
    delinquency_bands: list[int]


class ValidationConfig(BaseModel):
    version: str
    seed: int
    holdout_fraction: float
    cv_folds: int
    bootstrap_iterations: int
    confidence: float
    calibration_bins: int
    low_risk_deciles: int
    champion_selection: ChampionSelection
    fairness: FairnessConfig
    lda: LDAConfig
    lane_b: LaneBConfig


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
    return PolicyRules.model_validate(read_yaml("policy_v2.yaml"))


def policy_file_text() -> str:
    return (_rules_dir() / "policy_v2.yaml").read_text(encoding="utf-8")


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
def load_validation() -> ValidationConfig:
    return ValidationConfig.model_validate(read_yaml("validation.yaml"))


@lru_cache(maxsize=1)
def load_sanctions() -> SanctionsList:
    return SanctionsList.model_validate(read_yaml("reference/sanctions.yaml"))
