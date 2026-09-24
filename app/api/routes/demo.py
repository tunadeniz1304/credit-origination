"""Demo scenario loader (admin only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import require_roles
from app.core.security import Principal
from app.demo.seed import SCENARIOS, seed_demo

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])


@router.get("/scenarios")
def scenarios(
    user: Principal = Depends(
        require_roles("uzman", "kidemli_uzman", "komite", "model_yoneticisi")
    ),
) -> dict[str, Any]:
    return {
        "scenarios": [
            {"key": s.key, "title": s.title, "expected": list(s.expected)} for s in SCENARIOS
        ]
    }


@router.post("/seed")
def seed(user: Principal = Depends(require_roles())) -> dict[str, Any]:
    results = seed_demo()
    return {"results": results, "all_as_expected": all(r["as_expected"] for r in results)}
