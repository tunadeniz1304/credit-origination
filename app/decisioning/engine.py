"""Hybrid decision engine (pure function of the snapshot).

1. Policy rules (knock-out ``decline`` / ``refer``) from the versioned YAML.
2. Calibrated monotone LightGBM PD with SHAP, WoE scorecard points, and a
   shadow challenger score (never affects the outcome).
3. Limit determination: debt-service capacity, income multiple, product
   bounds and the PD band's limit factor → approvable amount. A reduced
   amount becomes a *conditional* approval (counter-offer); the gating PD is
   the PD of the structure actually offered.
4. Decision policy: refer rules or grey-zone PD → ``UZMAN_INCELEMESI``;
   PD ≤ approve cut-off → ``OTOMATIK_ONAY``; PD ≥ decline cut-off or any
   knock-out → ``OTOMATIK_RET`` (always with reason codes and, per KVKK
   art. 11, the right to object and obtain human review).
5. Risk-based price, then a **limit ↔ price fixed point**: the DSR is
   re-checked with the priced instalment (contract rate + BSMV + KKDF); if it
   breaches the product cap the amount is scaled down and re-priced until it
   fits (at most ``max_pricing_iterations`` rounds). A price above the legal
   cap declines with ``R18``.

Given the same snapshot, rule set and model versions the result is identical
(``POST /api/v1/decisions/{id}/replay``).
"""

from __future__ import annotations

import math
import time
from typing import Any

from pydantic import BaseModel, Field

from app.core.rules import PolicyRules, PricingConfig, load_reasons
from app.decisioning.features import annuity_factor, snapshot_hash, with_loan
from app.decisioning.models import ModelSet
from app.decisioning.rules_engine import RuleResult, evaluate_rules
from app.explainability.counterfactual import Counterfactual, search
from app.explainability.reasons import ReasonCode, build_reason_codes, render
from app.pricing.engine import PriceQuote, quote

REFERENCE_RATE = 0.45  # DSR reference rate; matches the training generator


class DecisionResult(BaseModel):
    outcome: str
    conditional: bool = False
    pd: float
    pd_requested: float
    risk_band: str
    score: dict[str, Any]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    rule_results: list[RuleResult] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)
    pricing: PriceQuote | None = None
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)
    challenger: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    feature_hash: str = ""
    offer_amount: float = 0.0
    offer_term: int = 0
    latency_ms: float = 0.0

    @property
    def reason_code_list(self) -> list[str]:
        return [r.code for r in self.reason_codes]


def _floor_thousand(value: float) -> float:
    return float(math.floor(value / 1000.0) * 1000.0) if value > 0 else 0.0


def _gate(
    snapshot: dict[str, Any], policy: PolicyRules, models: ModelSet, pricing_cfg: PricingConfig
) -> tuple[bool, float]:
    """Would this exact structure be auto-approved? (used by counterfactuals)."""
    rules = evaluate_rules(snapshot, policy)
    if any(r.fired for r in rules):
        return False, 1.0
    product = policy.product(snapshot.get("product", "IHTIYAC"))
    if snapshot["dsr"] > product.max_dsr:
        return False, 1.0
    pd = models.pd_model.predict(snapshot, explain=False).pd
    if pd > policy.decision.auto_approve_max_pd:
        return False, pd
    priced = quote(
        pd=pd,
        amount=snapshot["requested_amount"],
        term_months=snapshot["term_months"],
        product=snapshot.get("product", "IHTIYAC"),
        cfg=pricing_cfg,
        include_schedule=False,
    )
    real_dsr = priced_dsr(snapshot, priced.instalment)
    return (not priced.exceeds_cap and real_dsr <= product.max_dsr), pd


def priced_dsr(snapshot: dict[str, Any], instalment: float) -> float:
    """DSR with the actual priced instalment (taxes included), not the reference rate."""
    income = max(float(snapshot["monthly_income"]), 1.0)
    return (float(snapshot["existing_debt_service"]) + instalment) / income


def _outcome(
    gating_pd: float, *, declines: bool, refers: bool, can_offer: bool, policy: PolicyRules
) -> str:
    thresholds = policy.decision
    if declines or not can_offer:
        return "OTOMATIK_RET"
    if refers:
        return "UZMAN_INCELEMESI"
    if gating_pd >= thresholds.auto_decline_min_pd:
        return "OTOMATIK_RET"
    if gating_pd <= thresholds.auto_approve_max_pd:
        return "OTOMATIK_ONAY"
    return "UZMAN_INCELEMESI"


def decide(
    snapshot: dict[str, Any],
    *,
    policy: PolicyRules,
    models: ModelSet,
    pricing_cfg: PricingConfig,
    rule_set_version: str | None = None,
) -> DecisionResult:
    started = time.perf_counter()
    product_code = snapshot.get("product", "IHTIYAC")
    product = policy.product(product_code)
    thresholds = policy.decision
    income = max(float(snapshot["monthly_income"]), 1.0)
    requested = float(snapshot["requested_amount"])
    term = int(min(max(int(snapshot["term_months"]), product.min_term), product.max_term))
    term_adjusted = term != int(snapshot["term_months"])

    rules = evaluate_rules(snapshot, policy)
    declines = [r for r in rules if r.fired and r.action == "decline"]
    refers = [r for r in rules if r.fired and r.action == "refer"]

    requested_out = models.pd_model.predict(snapshot)
    scorecard = models.scorecard.score(snapshot)
    challenger_pd = models.challenger.predict(snapshot)

    # ---- limit determination
    existing = float(snapshot["existing_debt_service"])
    unit = annuity_factor(term, REFERENCE_RATE)
    capacity = max(0.0, product.max_dsr * income - existing) / unit
    ceiling = min(product.max_amount, income * product.income_multiplier)
    candidate = _floor_thousand(min(requested, capacity, ceiling))
    offer_snapshot = with_loan(
        snapshot, amount=candidate or requested, term=term, reference_rate=REFERENCE_RATE
    )
    offer_out = models.pd_model.predict(offer_snapshot, explain=False)
    band = policy.band_for(offer_out.pd)
    max_amount = _floor_thousand(min(capacity, ceiling * band.limit_factor))
    offer_amount = _floor_thousand(min(requested, max_amount))
    if offer_amount != candidate and offer_amount > 0:
        offer_snapshot = with_loan(
            snapshot, amount=offer_amount, term=term, reference_rate=REFERENCE_RATE
        )
        offer_out = models.pd_model.predict(offer_snapshot, explain=False)
    min_offer = max(product.min_amount, requested * thresholds.counter_offer_min_ratio)
    can_offer = offer_amount >= min_offer
    conditional = can_offer and (offer_amount < requested or term_adjusted)
    gating_pd = offer_out.pd if can_offer else requested_out.pd

    context = {**snapshot, "max_dsr": product.max_dsr}
    fired_dicts = [r.model_dump() for r in declines + refers]
    extra_reasons: list[str] = []
    outcome = _outcome(
        gating_pd, declines=bool(declines), refers=bool(refers), can_offer=can_offer, policy=policy
    )
    if not declines and not can_offer and band.limit_factor > 0:
        # otherwise the PD band itself explains the decline
        extra_reasons.append("R01_DSR_YUKSEK" if capacity < min_offer else "R09_TUTAR_GELIR")

    priced: PriceQuote | None = None
    iterations = 0
    real_dsr: float | None = None
    while outcome != "OTOMATIK_RET" and can_offer:
        iterations += 1
        priced = quote(
            pd=gating_pd,
            amount=offer_amount,
            term_months=term,
            product=product_code,
            cfg=pricing_cfg,
        )
        if priced.exceeds_cap:
            outcome = "OTOMATIK_RET"
            extra_reasons.append("R18_FAIZ_TAVANI")
            priced = None
            break
        real_dsr = priced_dsr(snapshot, priced.instalment)
        if real_dsr <= product.max_dsr or iterations >= thresholds.max_pricing_iterations:
            break
        # Scale the principal to the instalment the DSR cap still allows, re-price.
        affordable = max(0.0, product.max_dsr * income - existing)
        scaled = _floor_thousand(offer_amount * affordable / priced.instalment)
        offer_amount = min(scaled, offer_amount - 1000.0)
        if offer_amount < min_offer:
            can_offer = False
            outcome = "OTOMATIK_RET"
            extra_reasons.append("R01_DSR_YUKSEK")
            priced = None
            break
        offer_snapshot = with_loan(
            snapshot, amount=offer_amount, term=term, reference_rate=REFERENCE_RATE
        )
        offer_out = models.pd_model.predict(offer_snapshot, explain=False)
        gating_pd = offer_out.pd
        band = policy.band_for(gating_pd)
        outcome = _outcome(
            gating_pd,
            declines=bool(declines),
            refers=bool(refers),
            can_offer=can_offer,
            policy=policy,
        )
    if priced is not None and real_dsr is not None and real_dsr > product.max_dsr:
        # Did not converge within the iteration budget: never offer above the cap.
        outcome, priced, can_offer = "OTOMATIK_RET", None, False
        extra_reasons.append("R01_DSR_YUKSEK")
    conditional = can_offer and (offer_amount < requested or term_adjusted)
    if conditional and offer_amount < requested:
        # Explain the reduction with the DSR the *requested* structure would really have.
        at_request = quote(
            pd=requested_out.pd,
            amount=requested,
            term_months=term,
            product=product_code,
            cfg=pricing_cfg,
            include_schedule=False,
        )
        context["dsr"] = round(
            max(float(snapshot["dsr"]), priced_dsr(snapshot, at_request.instalment)), 4
        )

    approved = outcome == "OTOMATIK_ONAY"
    reasons = build_reason_codes(
        context=context,
        fired_rules=fired_dicts,
        shap_values=requested_out.shap,
        points_lost=scorecard.get("points_lost"),
        include_model=not approved or conditional,
        approved=approved,
    )
    if conditional and context["dsr"] > product.max_dsr:
        # The counter-offer's own condition: stated plainly, first, never as "improvement".
        reasons = [r for r in reasons if r.code not in ("R01_DSR_YUKSEK", "R27_DSR_SINIRDA")]
        if "R01_DSR_YUKSEK" not in extra_reasons:
            extra_reasons.insert(0, "R01_DSR_YUKSEK")
    known = {r.code for r in reasons}
    for code in reversed(extra_reasons):
        if code not in known:
            reasons.insert(
                0,
                ReasonCode(
                    code=code,
                    text=render(code, context),
                    source="rule",
                    kind="condition" if approved else "adverse",
                ),
            )
            known.add(code)
    reasons = reasons[: load_reasons().materiality.max_codes]

    counterfactuals: list[Counterfactual] = []
    if outcome != "OTOMATIK_ONAY" or conditional:
        counterfactuals = search(
            snapshot,
            approves=lambda s: _gate(s, policy, models, pricing_cfg),
            max_term=product.max_term,
            min_amount=product.min_amount,
            reference_rate=REFERENCE_RATE,
            min_ratio=thresholds.counter_offer_min_ratio,
        )

    shap_sorted = sorted(requested_out.shap.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return DecisionResult(
        outcome=outcome,
        conditional=conditional and outcome == "OTOMATIK_ONAY",
        pd=round(gating_pd, 5),
        pd_requested=requested_out.pd,
        risk_band=band.band if can_offer else policy.band_for(requested_out.pd).band,
        score=scorecard,
        reason_codes=reasons,
        rule_results=rules,
        limits={
            "requested_amount": requested,
            "dsr_capacity_amount": round(capacity, 2),
            "income_ceiling": round(ceiling, 2),
            "band_limit_factor": band.limit_factor,
            "max_approvable_amount": max_amount,
            "offer_amount": offer_amount if can_offer else 0.0,
            "term_months": term,
            "term_adjusted": term_adjusted,
            "max_dsr": product.max_dsr,
            "dsr_requested": snapshot["dsr"],
            "dsr_offer": round(real_dsr, 4) if real_dsr is not None else offer_snapshot["dsr"],
            "dsr_offer_reference": offer_snapshot["dsr"],
            "pricing_iterations": iterations,
        },
        pricing=priced,
        counterfactuals=counterfactuals,
        explanation={
            "base_value": requested_out.base_value,
            "raw_score": requested_out.raw_score,
            "shap": dict(shap_sorted),
        },
        challenger={
            "version": models.challenger.version,
            "pd": challenger_pd,
            "champion_pd": requested_out.pd,
            "delta": round(challenger_pd - requested_out.pd, 5),
            "would_approve": challenger_pd <= thresholds.auto_approve_max_pd,
        },
        versions={
            "rule_set": rule_set_version or policy.version,
            "model": models.pd_model.version,
            "scorecard": models.scorecard.version,
            "pricing": pricing_cfg.version,
        },
        feature_hash=snapshot_hash(snapshot),
        offer_amount=offer_amount
        if outcome == "OTOMATIK_ONAY" or outcome == "UZMAN_INCELEMESI"
        else 0.0,
        offer_term=term,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
