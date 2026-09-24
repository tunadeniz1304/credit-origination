"""Credit authority matrix and four-eyes (maker-checker) rules."""

from __future__ import annotations

from pydantic import BaseModel

from app.core.rules import AuthorityMatrix, load_authority
from app.core.security import CREDIT_AUTHORITY_RANK, Principal


class AuthorityCheck(BaseModel):
    required_role: str
    required_rank: int
    four_eyes: bool
    maker_rank: int
    maker_may_finalise: bool
    reasons: list[str]


def required_role(amount: float, pd: float, matrix: AuthorityMatrix | None = None) -> str:
    matrix = matrix or load_authority()
    for level in matrix.levels:
        if amount <= level.max_amount and pd <= level.max_pd:
            return level.role
    return matrix.levels[-1].role


def evaluate_authority(
    maker: Principal,
    *,
    amount: float,
    pd: float,
    is_override: bool,
    matrix: AuthorityMatrix | None = None,
) -> AuthorityCheck:
    matrix = matrix or load_authority()
    role = required_role(amount, pd, matrix)
    rank = CREDIT_AUTHORITY_RANK[role]
    reasons: list[str] = []
    cfg = matrix.four_eyes
    if amount > cfg.amount_threshold:
        reasons.append(f"tutar {amount:,.0f} > {cfg.amount_threshold:,.0f}")
    if pd > cfg.pd_threshold:
        reasons.append(f"PD {pd:.3f} > {cfg.pd_threshold:.3f}")
    if is_override and cfg.overrides:
        reasons.append("motor kararına aykırı (override)")
    four_eyes = bool(reasons)
    return AuthorityCheck(
        required_role=role,
        required_rank=rank,
        four_eyes=four_eyes,
        maker_rank=maker.authority_rank,
        maker_may_finalise=not four_eyes and maker.authority_rank >= rank,
        reasons=reasons,
    )


def checker_allowed(
    checker: Principal, maker_username: str, required_rank: int
) -> tuple[bool, str]:
    if checker.username == maker_username:
        return False, "dört göz ilkesi: talebi oluşturan kişi onaylayamaz"
    if checker.authority_rank < required_rank:
        return False, "onay yetkiniz bu tutar/risk bandı için yetersiz"
    return True, ""
