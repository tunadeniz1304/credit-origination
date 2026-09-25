"""Applicant-specific Turkish reason codes.

Sources, in priority order:

1. fired policy rules (their own reason codes),
2. the most adverse SHAP contributions of the PD model (features pushing the
   log-odds of default up), mapped through ``rules/reason_codes.yaml``,
3. scorecard points lost (shown next to each reason in the workbench).

A model reason is produced only when it is **material** — its SHAP
contribution and scorecard points lost exceed the thresholds in
``rules/reason_codes.yaml`` — **and** the applicant's value lies in the
adverse range of that feature (``adverse_when``): 52 months of tenure on a
low-PD file is not "short tenure". At most ``max_codes`` codes are returned,
rules first, then model reasons by contribution. On approved files the same
signals are phrased as improvement areas (``kind="improvement"``), never in
decline language.

Each code is rendered with the applicant's own values, e.g.
``R01_DSR_YUKSEK: "Aylık borç ödemelerinizin gelirinize oranı %58,0 ile
politika sınırı olan %50,0 seviyesinin üzerinde."``
"""

from __future__ import annotations

import string
from typing import Any

from pydantic import BaseModel

from app.core.rules import ReasonCatalog, load_reasons
from app.decisioning.expressions import evaluate

PERCENT_FIELDS = {"dsr", "max_dsr", "gambling_share", "savings_rate", "bureau_utilisation"}


class ReasonCode(BaseModel):
    code: str
    text: str
    source: str  # rule | model
    feature: str | None = None
    contribution: float | None = None  # SHAP log-odds contribution
    points_lost: float | None = None
    kind: str = "adverse"  # adverse | condition | improvement


def _fmt_number(value: float, decimals: int) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_value(field: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, int | float):
        if field in PERCENT_FIELDS:
            return f"%{_fmt_number(float(value) * 100, 1)}"
        if field in ("income_cv", "loan_to_income"):
            return _fmt_number(float(value), 2)
        if float(value).is_integer():
            return _fmt_number(float(value), 0)
        return _fmt_number(float(value), 2)
    return str(value)


def render(
    code: str,
    context: dict[str, Any],
    catalog: ReasonCatalog | None = None,
    *,
    improvement: bool = False,
) -> str:
    catalog = catalog or load_reasons()
    template = catalog.codes.get(code, code)
    if improvement:
        template = catalog.improvements.get(code, template)
    names = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    values = {name: format_value(name, context.get(name)) for name in names}
    return template.format(**values)


def _within_cap(context: dict[str, Any]) -> bool:
    dsr, cap = context.get("dsr"), context.get("max_dsr")
    return dsr is not None and cap is not None and float(dsr) <= float(cap)


def is_adverse(feature: str, context: dict[str, Any], catalog: ReasonCatalog) -> bool:
    """True when the applicant's value is in the adverse range for ``feature``."""
    condition = catalog.adverse_when.get(feature)
    if condition is None:
        return True
    try:
        return bool(evaluate(condition, context))
    except Exception:  # unknown variable: never invent a reason
        return False


def is_material(contribution: float, points_lost: float | None, catalog: ReasonCatalog) -> bool:
    rules = catalog.materiality
    if contribution < rules.min_shap:
        return False
    return points_lost is None or points_lost >= rules.min_points_lost


def build_reason_codes(
    *,
    context: dict[str, Any],
    fired_rules: list[dict[str, Any]],
    shap_values: dict[str, float],
    points_lost: dict[str, float] | None = None,
    include_model: bool = True,
    approved: bool = False,
    catalog: ReasonCatalog | None = None,
) -> list[ReasonCode]:
    catalog = catalog or load_reasons()
    points_lost = points_lost or {}
    limit = catalog.materiality.max_codes
    reasons: list[ReasonCode] = []
    seen: set[str] = set()
    for rule in fired_rules:
        code = rule["reason"]
        if code in seen:
            continue
        seen.add(code)
        reasons.append(ReasonCode(code=code, text=render(code, context, catalog), source="rule"))
    if include_model:
        adverse = sorted(
            ((f, v) for f, v in shap_values.items() if v > 0), key=lambda kv: kv[1], reverse=True
        )
        for feature, contribution in adverse:
            if len(reasons) >= limit:
                break
            code = catalog.feature_reasons.get(feature)
            if not code or code in seen:
                continue
            thin_file = feature == "bureau_score" and context.get("bureau_hit") == 0
            if thin_file:
                code = "R20_INCE_DOSYA"
            elif code == "R01_DSR_YUKSEK" and _within_cap(context):
                code = "R27_DSR_SINIRDA"  # high but inside the cap: never claim "above"
            if code in seen:
                continue
            lost = points_lost.get(feature)
            if not is_material(contribution, lost, catalog):
                continue
            if not thin_file and not is_adverse(feature, context, catalog):
                continue
            seen.add(code)
            reasons.append(
                ReasonCode(
                    code=code,
                    text=render(code, context, catalog, improvement=approved),
                    source="model",
                    feature=feature,
                    contribution=round(contribution, 4),
                    points_lost=lost,
                    kind="improvement" if approved else "adverse",
                )
            )
    return reasons[:limit] if len(reasons) > limit else reasons
