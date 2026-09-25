"""Risk-based pricing (FICO pricing-optimisation style, simplified).

* Expected loss ``EL = PD × LGD × EAD`` (annualised on the average exposure).
* Economic capital per unit of exposure from the Basel IRB retail formula
  ``K = LGD·[N((G(PD) + √R·G(0.999)) / √(1−R)) − PD]`` with the PD-dependent
  "other retail" correlation
  ``R = 0.03·w + 0.16·(1 − w)``, ``w = (1 − e^(−35·PD)) / (1 − e^(−35))``.
* ``RAROC(r) = (r − funding − opex − PD·LGD) / K``; the minimum contractual
  rate that meets the product's target RAROC is solved with
  ``scipy.optimize.brentq`` and floored at the product minimum.
* BSMV and KKDF are charged on interest; the TCMB contractual-rate cap is
  enforced (quotes above the cap are flagged, the decision engine then
  declines or proposes another structure).
* The annual cost rate (yıllık maliyet oranı) is the IRR of the customer's
  cash flows including taxes and the upfront fee.

All parameters come from ``rules/pricing.yaml``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, Field
from scipy.optimize import brentq
from scipy.stats import norm

from app.core.rules import PricingConfig, load_pricing

_CENT = Decimal("0.01")


class ScheduleRow(BaseModel):
    period: int
    opening_balance: float
    instalment: float
    interest: float
    kkdf: float
    bsmv: float
    principal: float
    closing_balance: float


class PriceQuote(BaseModel):
    product: str
    amount: float
    term_months: int
    pd: float
    lgd: float
    expected_loss_annual: float
    capital_ratio: float
    asset_correlation: float
    raroc: float
    target_raroc: float
    annual_rate: float  # contractual (akdi) annual rate, before taxes
    monthly_rate: float
    gross_monthly_rate: float  # incl. BSMV + KKDF
    instalment: float
    total_payment: float
    total_interest: float
    total_taxes: float
    upfront_fee: float
    apr: float  # annual cost rate, effective (IRR incl. taxes and fees)
    apr_nominal: float  # monthly cost rate × 12
    legal_cap_annual: float
    exceeds_cap: bool
    config_version: str
    schedule: list[ScheduleRow] = Field(default_factory=list)


def retail_correlation(pd: float, cfg: PricingConfig | None = None) -> float:
    """Basel IRB asset correlation for "other retail" exposures (falls from 16 % to 3 %)."""
    c = (cfg or load_pricing()).correlation
    weight = (1 - math.exp(-c.k * pd)) / (1 - math.exp(-c.k))
    return c.r_min * weight + c.r_max * (1 - weight)


def irb_capital(
    pd: float,
    lgd: float,
    correlation: float,
    confidence: float,
    *,
    pd_floor: float = 0.0003,
    capital_floor: float = 0.0,
) -> float:
    """Unexpected-loss capital per unit of exposure (Basel IRB, no maturity adjustment)."""
    pd = min(max(pd, pd_floor), 0.9999)
    g_pd = norm.ppf(pd)
    g_conf = norm.ppf(confidence)
    conditional = norm.cdf((g_pd + math.sqrt(correlation) * g_conf) / math.sqrt(1 - correlation))
    return max(lgd * (conditional - pd), capital_floor)


def capital_ratio(pd: float, lgd: float, cfg: PricingConfig) -> float:
    pd_eff = max(pd, cfg.pd_floor)
    return irb_capital(
        pd_eff,
        lgd,
        retail_correlation(pd_eff, cfg),
        cfg.confidence_level,
        pd_floor=cfg.pd_floor,
        capital_floor=cfg.capital_floor,
    )


def raroc(rate: float, pd: float, product: str, cfg: PricingConfig) -> float:
    p = cfg.product(product)
    capital = capital_ratio(pd, p.lgd, cfg)
    margin = rate - p.funding_cost_annual - p.opex_annual - pd * p.lgd
    return margin / capital


def minimum_rate(pd: float, product: str, cfg: PricingConfig) -> float:
    p = cfg.product(product)
    target = p.target_raroc

    def gap(rate: float) -> float:
        return raroc(rate, pd, product, cfg) - target

    rate = brentq(gap, 0.0, 5.0, xtol=1e-7)
    return max(rate, p.min_rate_annual)


def _money(value: Decimal) -> float:
    return float(value.quantize(_CENT, rounding=ROUND_HALF_UP))


def build_taxed_schedule(
    amount: float, term_months: int, annual_rate: float, cfg: PricingConfig
) -> tuple[list[ScheduleRow], Decimal]:
    """Decimal annuity where each instalment covers interest + KKDF + BSMV."""
    taxes = cfg.taxes
    kkdf, bsmv = Decimal(str(taxes.get("kkdf", 0.0))), Decimal(str(taxes.get("bsmv", 0.0)))
    i = Decimal(str(annual_rate)) / 12
    gross = i * (1 + kkdf + bsmv)
    p = Decimal(str(amount))
    n = max(1, term_months)
    if gross == 0:
        instalment = (p / n).quantize(_CENT, rounding=ROUND_HALF_UP)
    else:
        factor = (1 + gross) ** n
        instalment = (p * gross * factor / (factor - 1)).quantize(_CENT, rounding=ROUND_HALF_UP)
    rows: list[ScheduleRow] = []
    balance = p
    for period in range(1, n + 1):
        interest = (balance * i).quantize(_CENT, rounding=ROUND_HALF_UP)
        tax_kkdf = (interest * kkdf).quantize(_CENT, rounding=ROUND_HALF_UP)
        tax_bsmv = (interest * bsmv).quantize(_CENT, rounding=ROUND_HALF_UP)
        payment = instalment
        principal = payment - interest - tax_kkdf - tax_bsmv
        if period == n or principal > balance:
            principal = balance
            payment = principal + interest + tax_kkdf + tax_bsmv
        closing = balance - principal
        rows.append(
            ScheduleRow(
                period=period,
                opening_balance=_money(balance),
                instalment=_money(payment),
                interest=_money(interest),
                kkdf=_money(tax_kkdf),
                bsmv=_money(tax_bsmv),
                principal=_money(principal),
                closing_balance=_money(closing),
            )
        )
        balance = closing
    return rows, instalment


def annual_cost_rate(amount: float, fee: float, payments: list[float]) -> tuple[float, float]:
    """(effective annual, nominal annual) cost rates from the monthly IRR."""
    net = amount - fee

    def npv(r: float) -> float:
        return -net + sum(pmt / (1 + r) ** (t + 1) for t, pmt in enumerate(payments))

    monthly = brentq(npv, 1e-9, 1.0, xtol=1e-10)
    return (1 + monthly) ** 12 - 1, monthly * 12


def quote(
    *,
    pd: float,
    amount: float,
    term_months: int,
    product: str = "IHTIYAC",
    cfg: PricingConfig | None = None,
    rate_override: float | None = None,
    include_schedule: bool = True,
) -> PriceQuote:
    cfg = cfg or load_pricing()
    p = cfg.product(product)
    rate = rate_override if rate_override is not None else minimum_rate(pd, product, cfg)
    rate = round(rate, 4)
    rows, instalment = build_taxed_schedule(amount, term_months, rate, cfg)
    fee = round(amount * p.upfront_fee_rate * (1 + cfg.taxes.get("bsmv", 0.0)), 2)
    payments = [row.instalment for row in rows]
    apr, apr_nominal = annual_cost_rate(amount, fee, payments)
    total_interest = sum(row.interest for row in rows)
    total_taxes = sum(row.kkdf + row.bsmv for row in rows)
    capital = capital_ratio(pd, p.lgd, cfg)
    return PriceQuote(
        product=product,
        amount=round(amount, 2),
        term_months=term_months,
        pd=round(pd, 5),
        lgd=p.lgd,
        expected_loss_annual=round(pd * p.lgd * amount, 2),
        capital_ratio=round(capital, 4),
        asset_correlation=round(retail_correlation(max(pd, cfg.pd_floor), cfg), 5),
        raroc=round(raroc(rate, pd, product, cfg), 4),
        target_raroc=p.target_raroc,
        annual_rate=rate,
        monthly_rate=round(rate / 12, 6),
        gross_monthly_rate=round(rate / 12 * cfg.tax_multiplier, 6),
        instalment=float(instalment),
        total_payment=round(sum(payments), 2),
        total_interest=round(total_interest, 2),
        total_taxes=round(total_taxes, 2),
        upfront_fee=fee,
        apr=round(apr, 4),
        apr_nominal=round(apr_nominal, 4),
        legal_cap_annual=cfg.legal_cap_annual,
        exceeds_cap=rate > cfg.legal_cap_annual,
        config_version=cfg.version,
        schedule=rows if include_schedule else [],
    )
