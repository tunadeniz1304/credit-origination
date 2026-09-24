"""Counterfactual ("what would get me approved") explanations.

A constrained search over the features an applicant can actually change —
requested amount, term and closing part of the existing debt — never over
immutable or protected attributes. Each candidate is re-scored by the same
deterministic engine; the smallest changes that turn the outcome into an
approval are returned as plain-Turkish suggestions. ``dice-ml`` could be
plugged in as an optional extra; the built-in search is exact for this
low-dimensional, monotone space.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from app.decisioning.features import with_loan


class Counterfactual(BaseModel):
    kind: str  # amount | term | amount_term | debt
    changes: dict[str, float]
    pd: float
    text: str


def _tl(value: float) -> str:
    text = f"{value:,.0f}"
    return text.replace(",", ".") + " TL"


def search(
    snapshot: dict[str, Any],
    *,
    approves: Callable[[dict[str, Any]], tuple[bool, float]],
    max_term: int,
    min_amount: float,
    reference_rate: float,
    min_ratio: float = 0.3,
    limit: int = 3,
) -> list[Counterfactual]:
    """Return up to ``limit`` minimal approving changes (smallest first)."""
    amount = float(snapshot["requested_amount"])
    term = int(snapshot["term_months"])
    found: list[Counterfactual] = []

    # 1) Lower amount at the same term (5% steps).
    for step in range(1, 15):
        candidate = round(amount * (1 - 0.05 * step), -3)
        if candidate < max(min_amount, amount * min_ratio):
            break
        ok, pd = approves(
            with_loan(snapshot, amount=candidate, term=term, reference_rate=reference_rate)
        )
        if ok:
            found.append(
                Counterfactual(
                    kind="amount",
                    changes={"requested_amount": candidate},
                    pd=pd,
                    text=f"Kredi tutarını {_tl(candidate)} seviyesine düşürmeniz halinde başvurunuzun "
                    "onaylanma olasılığı yüksektir.",
                )
            )
            break

    # 2) Longer term at the same amount.
    for candidate_term in range(term + 6, max_term + 1, 6):
        ok, pd = approves(
            with_loan(snapshot, amount=amount, term=candidate_term, reference_rate=reference_rate)
        )
        if ok:
            found.append(
                Counterfactual(
                    kind="term",
                    changes={"term_months": candidate_term},
                    pd=pd,
                    text=f"Vadeyi {candidate_term} aya çıkarmanız halinde aylık taksitiniz düşer ve "
                    "onay olasılığı yükselir.",
                )
            )
            break

    # 3) Close part of existing debt (reduces existing debt service).
    existing = float(snapshot.get("existing_debt_service", 0.0))
    if existing > 0:
        for share in (0.25, 0.5, 0.75, 1.0):
            reduced = dict(snapshot)
            reduced["existing_debt_service"] = round(existing * (1 - share), 2)
            reduced["existing_dsr"] = round(
                reduced["existing_debt_service"] / max(float(snapshot["monthly_income"]), 1), 4
            )
            reduced = with_loan(reduced, amount=amount, term=term, reference_rate=reference_rate)
            ok, pd = approves(reduced)
            if ok:
                found.append(
                    Counterfactual(
                        kind="debt",
                        changes={"existing_debt_service": reduced["existing_debt_service"]},
                        pd=pd,
                        text=f"Mevcut aylık kredi/kart ödemelerinizin yaklaşık %{int(share * 100)}'ini "
                        "kapatmanız halinde borç servis oranınız politika sınırına iner.",
                    )
                )
                break

    # 4) Combined: moderate amount cut + longer term.
    if not found:
        for candidate_term in range(term + 12, max_term + 1, 12):
            for step in (0.1, 0.2, 0.3):
                candidate = round(amount * (1 - step), -3)
                if candidate < max(min_amount, amount * min_ratio):
                    continue
                ok, pd = approves(
                    with_loan(
                        snapshot,
                        amount=candidate,
                        term=candidate_term,
                        reference_rate=reference_rate,
                    )
                )
                if ok:
                    found.append(
                        Counterfactual(
                            kind="amount_term",
                            changes={"requested_amount": candidate, "term_months": candidate_term},
                            pd=pd,
                            text=f"Tutarı {_tl(candidate)} ve vadeyi {candidate_term} ay olarak "
                            "güncellemeniz halinde onay olasılığı yüksektir.",
                        )
                    )
                    return found[:limit]
    return found[:limit]
