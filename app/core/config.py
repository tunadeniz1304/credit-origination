"""Application settings and typed business rules.

`Settings` is the single source of environment configuration (real
environment first, then ``Anil2/.env``, then ``../.env``). `PipelineRules` is
the typed view of ``config/config.json``; the versioned YAML rule files under
``rules/`` are loaded through :mod:`app.core.rules`. Both are cached
process-wide.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import DocumentRequirement

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

# Order matters: pydantic-settings lets later files override earlier ones, so the
# project-local ``.env`` wins over the parent directory's ``.env``.
_ENV_FILES = (str(PROJECT_ROOT.parent / ".env"), str(PROJECT_ROOT / ".env"))

DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEV_JWT_FALLBACK = "dev-only-insecure-jwt-secret-change-me"


def _dotenv_enabled() -> tuple[str, ...] | None:
    """Tests can disable dotenv loading entirely via ``ANIL2_NO_DOTENV=1``."""
    return None if os.environ.get("ANIL2_NO_DOTENV") == "1" else _ENV_FILES


class Settings(BaseSettings):
    """Environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Anil2 Kredi Tahsis Platformu"
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # Persistence. Relative sqlite paths resolve against the project root.
    database_url: str = "sqlite:///data/anil2.db"
    db_auto_create: bool = True
    # SQLite: one writer transaction at a time per process (see app.db.session).
    db_writer_lock_timeout_seconds: float = 60.0
    db_busy_timeout_ms: int = 30_000
    # A request session keeps its connection until its background task has run,
    # so the pool must cover requests + inline pipelines (SQLite connections are cheap).
    db_pool_size: int = 20
    db_max_overflow: int = 60
    audit_lock_retries: int = 5
    audit_retry_base_seconds: float = 0.05

    # Queue infrastructure.
    redis_url: str = "redis://localhost:6379/0"
    result_backend: str | None = None
    task_queue_backend: Literal["auto", "celery", "inline"] = "auto"

    # ---- LLM contract (see README "LLM modes") ----
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "LLM_API_KEY", "DEEPSEEK_API_KEY", "GATEWAY_API_KEY", "OPENAI_API_KEY"
        ),
    )
    llm_base_url: str = Field(
        default=DEFAULT_LLM_BASE_URL,
        validation_alias=AliasChoices(
            "LLM_BASE_URL", "DEEPSEEK_BASE_URL", "GATEWAY_BASE_URL", "OPENAI_BASE_URL"
        ),
    )
    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL, validation_alias=AliasChoices("LLM_MODEL", "DEEPSEEK_MODEL")
    )
    llm_mode: Literal["auto", "live", "demo"] = "auto"
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2000
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-sonnet-5"

    # ---- Security ----
    jwt_secret: SecretStr = SecretStr(DEV_JWT_FALLBACK)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480
    pii_encryption_key: SecretStr = SecretStr("")
    blind_index_key: SecretStr = SecretStr("")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )
    rate_limit_default: str = "300/minute"
    rate_limit_login: str = "20/minute"
    rate_limit_submit: str = "60/minute"
    rate_limit_enabled: bool = True
    rate_limit_register: str = "5/hour"
    # Demo users: on by default in dev/test, off in prod unless explicitly enabled.
    seed_demo_users: bool | None = Field(
        default=None, validation_alias=AliasChoices("SEED_DEMO_USERS", "DEMO_USERS_ENABLED")
    )
    demo_password: SecretStr = SecretStr("Demo123!")
    registration_enabled: bool = True
    captcha_provider: Literal["none", "hook"] = "none"
    # Browser sessions: HttpOnly cookie + signed double-submit CSRF token.
    session_cookie_name: str = "anil2_session"
    csrf_cookie_name: str = "anil2_csrf"
    csrf_header_name: str = "X-CSRF-Token"
    cookie_secure: bool = True

    # ---- Uploads ----
    max_upload_mb: float = 10.0
    allowed_upload_types: list[str] = Field(
        default_factory=lambda: ["application/pdf", "image/png", "image/jpeg", "text/plain"]
    )
    clamav_host: str = ""

    # ---- External (mock) providers ----
    kkb_base_url: str = "http://mock-kkb:8001/api/v1"
    edevlet_base_url: str = "http://mock-edevlet:8002/api/v1"
    gib_base_url: str = "http://mock-gib:8003/api/v1"
    openbanking_base_url: str = "http://mock-gecit:8004/api/v1"
    api_timeout_seconds: float = 5.0
    mock_external: bool = True
    circuit_failure_threshold: int = 3
    circuit_reset_timeout_seconds: float = 30.0
    circuit_state_backend: Literal["auto", "memory", "redis"] = "auto"
    circuit_half_open_max_calls: int = 1  # probes admitted while HALF_OPEN
    fault_injection_rate: float = 0.0  # chaos testing of the KKB mock (0..1)

    # ---- Notifications ----
    webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 1025
    smtp_from: str = "kredi@anil2.local"

    # ---- Data retention (KVKK) ----
    retention_days_rejected: int = 180

    # ---- Artifact directories (relative to project root) ----
    report_output_dir: str = "data/reports"
    result_store_dir: str = "data/results"
    upload_dir: str = "data/uploads"
    model_dir: str = "artifacts/models"
    rules_dir: str = "rules"

    # RAG chunking.
    document_chunk_size: int = 800
    document_chunk_overlap: int = 80

    @field_validator("llm_base_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/") or DEFAULT_LLM_BASE_URL

    @model_validator(mode="after")
    def _environment_defaults(self) -> Settings:
        if self.seed_demo_users is None:
            self.seed_demo_users = self.app_env != "prod"
        return self

    @property
    def demo_mode(self) -> bool:
        return bool(self.seed_demo_users)

    # ---- Derived helpers ----
    @property
    def result_backend_url(self) -> str:
        return self.result_backend or self.redis_url

    def _abs(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def report_dir(self) -> Path:
        return self._abs(self.report_output_dir)

    @property
    def result_dir(self) -> Path:
        return self._abs(self.result_store_dir)

    @property
    def uploads_dir(self) -> Path:
        return self._abs(self.upload_dir)

    @property
    def models_path(self) -> Path:
        return self._abs(self.model_dir)

    @property
    def rules_path(self) -> Path:
        return self._abs(self.rules_dir)

    @property
    def sqlalchemy_url(self) -> str:
        """Database URL with relative sqlite paths anchored at the project root."""
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith("sqlite:////"):
            raw = url[len(prefix) :]
            if raw and raw != ":memory:" and not Path(raw).is_absolute():
                return prefix + str((PROJECT_ROOT / raw).as_posix())
        return url

    @property
    def llm_key_present(self) -> bool:
        return bool(self.llm_api_key.get_secret_value().strip())

    @property
    def llm_effective_mode(self) -> Literal["live", "demo"]:
        """Resolve ``auto`` against key presence (``live`` without a key is invalid)."""
        if self.llm_mode == "demo":
            return "demo"
        if self.llm_mode == "live":
            return "live"
        return "live" if self.llm_key_present else "demo"

    @property
    def llm_base_url_host(self) -> str:
        return urlparse(self.llm_base_url).hostname or self.llm_base_url

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def using_dev_jwt_secret(self) -> bool:
        return self.jwt_secret.get_secret_value() == DEV_JWT_FALLBACK


class LLMConfigurationError(RuntimeError):
    """Raised at start-up when ``LLM_MODE=live`` is forced without an API key."""


def validate_llm_settings(settings: Settings) -> None:
    """Fail fast when live mode is forced but no key is available."""
    if settings.llm_mode == "live" and not settings.llm_key_present:
        raise LLMConfigurationError(
            "LLM_MODE=live but no API key found (LLM_API_KEY / DEEPSEEK_API_KEY / "
            "GATEWAY_API_KEY / OPENAI_API_KEY)."
        )


class ApplicationRules(BaseModel):
    default_currency: str = "TRY"


class DocumentPolicy(BaseModel):
    """Mandatory document list per employment type; order matters for letters."""

    required_documents: list[DocumentRequirement]
    self_employed_documents: list[DocumentRequirement] = Field(default_factory=list)


class CommitteeRules(BaseModel):
    """Legacy thresholds kept for the transparent committee factor table."""

    min_kbb_score: int = 1100
    max_debt_service_ratio: float = 0.5
    max_loan_to_income_multiplier: float = 6.0
    max_term_months: int = 60


class PipelineRules(BaseModel):
    """Typed, versioned view of ``config/config.json``."""

    version: str = "1"
    application: ApplicationRules
    document_policy: DocumentPolicy
    committee: CommitteeRules


@lru_cache(maxsize=4)
def _load_rules_raw(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
        return data


@lru_cache(maxsize=1)
def load_pipeline_rules(path: Path | None = None) -> PipelineRules:
    """Load and validate the business rules from ``config/config.json``."""
    return PipelineRules.model_validate(_load_rules_raw(path) if path else _load_rules_raw())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached Settings."""
    return Settings(_env_file=_dotenv_enabled())  # type: ignore[call-arg]


def env_has(env_name: str) -> bool:
    """True when the environment variable is set and non-empty."""
    value = os.environ.get(env_name, "")
    return bool(value and value.strip())
