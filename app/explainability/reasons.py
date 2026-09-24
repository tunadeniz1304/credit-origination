"""Applicant-specific Turkish reason codes.

Sources, in priority order:

1. fired policy rules (their own reason codes),
2. the most adverse SHAP contributions of the PD model (features pushing the
   log-odds of default up), mapped through ``rules/reason_codes.yaml``,
3. scorecard points lost (shown next to each reason in the workbench).

Each code is rendered with the applicant's own values, e.g.
``R01_DSR_YUKSEK: "Aylık borç ödemelerinizin gelirinize oranı %58,0 ile
politika sınırı olan %50,0 seviyesinin üzerinde."``
"""

from __future__ import annotations

import string
from typing import Any

from pydantic import BaseModel

from app.core.rules import ReasonCatalog, load_reasons

PERCENT_FIELDS = {"dsr", "max_dsr", "gambling_share", "savings_rate", "bureau_utilisation"}
MAX_MODEL_REASONS = 4


class ReasonCode(BaseModel):
    code: str
    text: str
    source: str  # rule | model
    feature: str | None = None
    contribution: float | None = None  # SHAP log-odds contribution
    points_lost: float | None = None


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


def render(code: str, context: dict[str, Any], catalog: ReasonCatalog | None = None) -> str:
    catalog = catalog or load_reasons()
    template = catalog.codes.get(code, code)
    names = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    values = {name: format_value(name, context.get(name)) for name in names}
    return template.format(**values)


def build_reason_codes(
    *,
    context: dict[str, Any],
    fired_rules: list[dict[str, Any]],
    shap_values: dict[str, float],
    points_lost: dict[str, float] | None = None,
    include_model: bool = True,
    catalog: ReasonCatalog | None = None,
) -> list[ReasonCode]:
    catalog = catalog or load_reasons()
    points_lost = points_lost or {}
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
        added = 0
        for feature, contribution in adverse:
            code = catalog.feature_reasons.get(feature)
            if not code or code in seen:
                continue
            if feature == "bureau_score" and context.get("bureau_hit") == 0:
                code = "R20_INCE_DOSYA"
                if code in seen:
                    continue
            seen.add(code)
            reasons.append(
                ReasonCode(
                    code=code,
                    text=render(code, context, catalog),
                    source="model",
                    feature=feature,
                    contribution=round(contribution, 4),
                    points_lost=points_lost.get(feature),
                )
            )
            added += 1
            if added >= MAX_MODEL_REASONS:
                break
    return reasons
