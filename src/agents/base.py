"""Shared plumbing for pipeline agents: cached config access + a logger."""
from __future__ import annotations

import json
import os
from functools import lru_cache

from src.logger import get_logger


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache ``config/config.json`` (project root is two levels up)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "config", "config.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class AgentBase:
    """Base class exposing a named logger and cached configuration."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(f"agent.{name}")

    @property
    def config(self) -> dict:
        return load_config()
