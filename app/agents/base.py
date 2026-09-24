"""Shared plumbing for pipeline agents: typed rules access + a logger."""

from __future__ import annotations

from app.core.config import PipelineRules, load_pipeline_rules
from app.core.logging import get_logger


class AgentBase:
    """Base class exposing a named logger and the typed business rules."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @property
    def rules(self) -> PipelineRules:
        return load_pipeline_rules()
