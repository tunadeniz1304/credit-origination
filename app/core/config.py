"""Application settings and typed business rules.

`Settings` is the Pydantic-backed environment configuration (reads `.env`),
while `PipelineRules` is the typed view of the business rules shipped in
``config/config.json``. Both are cached process-wide.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import DocumentRequirement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


class Settings(BaseSettings):
    """Environment-driven configuration (``.env`` + real environment)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Akıllı Kredi Operasyon Ajanı"
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # Queue infrastructure.
    redis_url: str = "redis://localhost:6379/0"
    result_backend: str | None = None  # defaults to redis_url when unset
    task_queue_backend: Literal["auto", "celery", "inline"] = "auto"

    # Optional real LLM credentials; when absent a deterministic mock is used.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Mock external API base URLs (overridable; mock clients ignore them).
    kkb_base_url: str = "http://mock-kkb:8001/api/v1"
    edevlet_base_url: str = "http://mock-edevlet:8002/api/v1"
    api_timeout_seconds: float = 5.0
    api_latency_range_ms: tuple[int, int] = (20, 120)
    # When true, external clients run over an in-process mock HTTP transport
    # (deterministic payloads, zero network); when false they hit the base URLs.
    mock_external: bool = True
    circuit_failure_threshold: int = 3
    circuit_reset_timeout_seconds: float = 30.0

    # Artifact directories (relative to project root).
    report_output_dir: str = "data/reports"
    result_store_dir: str = "data/results"
    upload_dir: str = "data/uploads"

    # RAG chunking.
    document_chunk_size: int = 800
    document_chunk_overlap: int = 80

    @property
    def result_backend_url(self) -> str:
        return self.result_backend or self.redis_url

    @property
    def report_dir(self) -> Path:
        return PROJECT_ROOT / self.report_output_dir

    @property
    def result_dir(self) -> Path:
        return PROJECT_ROOT / self.result_store_dir

    @property
    def uploads_dir(self) -> Path:
        return PROJECT_ROOT / self.upload_dir


class ApiServiceConfig(BaseModel):
    """Per-provider HTTP client configuration."""

    base_url: str
    timeout_seconds: float = 5.0
    latency_ms: tuple[int, int] = (20, 120)


class ApiRules(BaseModel):
    kkb: ApiServiceConfig
    edevlet: ApiServiceConfig


class ApplicationRules(BaseModel):
    default_currency: str = "TRY"
    pipeline_run_delay_seconds: float = 0.0


class DocumentPolicy(BaseModel):
    """Mandatory document list; order matters for the draft template."""

    required_documents: list[DocumentRequirement]


class CommitteeRules(BaseModel):
    """Thresholds applied by the credit committee."""

    min_kbb_score: int = 1100
    max_debt_to_income_ratio: float = 0.5
    max_loan_to_income_multiplier: float = 6.0
    max_term_months: int = 60


class PipelineRules(BaseModel):
    """Typed, versioned view of ``config/config.json``."""

    application: ApplicationRules
    document_policy: DocumentPolicy
    api: ApiRules
    committee: CommitteeRules


@lru_cache(maxsize=1)
def _load_rules_raw(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_pipeline_rules(path: Path | None = None) -> PipelineRules:
    """Load and validate the business rules from ``config/config.json``."""
    return PipelineRules.model_validate(_load_rules_raw(path) if path else _load_rules_raw())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached Settings."""
    return Settings()


def env_has(env_name: str) -> bool:
    """True when the environment variable is set and non-empty."""
    value = os.environ.get(env_name, "")
    return bool(value and value.strip())
