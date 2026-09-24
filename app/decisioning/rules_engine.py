"""Knock-out / referral policy rules over the feature snapshot."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.core.rules import PolicyRules
from app.decisioning.expressions import evaluate


class RuleResult(BaseModel):
    id: str
    description: str
    action: str
    reason: str
    fired: bool
    error: str | None = None


def rule_context(snapshot: dict[str, Any], policy: PolicyRules) -> dict[str, Any]:
    product = policy.product(snapshot.get("product", "IHTIYAC"))
    return {
        **snapshot,
        "max_dsr": product.max_dsr,
        "min_document_confidence": policy.decision.min_document_confidence,
    }


def evaluate_rules(snapshot: dict[str, Any], policy: PolicyRules) -> list[RuleResult]:
    ctx = rule_context(snapshot, policy)
    results: list[RuleResult] = []
    for rule in policy.rules:
        try:
            fired = bool(evaluate(rule.when, ctx))
            error = None
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            # A broken rule must never silently approve: treat as a referral.
            fired, error = rule.action == "refer", f"{type(exc).__name__}: {exc}"
        results.append(
            RuleResult(
                id=rule.id,
                description=rule.description,
                action=rule.action,
                reason=rule.reason,
                fired=fired,
                error=error,
            )
        )
    return results
