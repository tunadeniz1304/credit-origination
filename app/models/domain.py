"""Shared domain value objects (pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DocumentRequirement(BaseModel):
    """One mandatory document from the policy, with a Turkish description."""

    model_config = ConfigDict(frozen=True)

    code: str
    description: str
