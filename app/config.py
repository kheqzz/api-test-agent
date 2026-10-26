"""Application settings, loaded once from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration."""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    batch_size: int
    max_endpoints: int
    rate_limit_rps: float
    request_timeout: float
    slow_threshold_ms: float
    llm_max_retries: int
    reports_db: str
    generated_tests_dir: str
    mount_demo: bool
    restrict_targets: bool
    allowed_target_hosts: tuple[str, ...]


settings = Settings(
    llm_api_key=os.getenv("LLM_API_KEY", ""),
    llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
    llm_model=os.getenv("LLM_MODEL", "your-model-name"),
    batch_size=max(1, _env_int("AGENT_BATCH_SIZE", 4)),
    max_endpoints=max(1, _env_int("AGENT_MAX_ENDPOINTS", 30)),
    rate_limit_rps=max(0.1, _env_float("AGENT_RATE_LIMIT_RPS", 5.0)),
    request_timeout=max(1.0, _env_float("AGENT_REQUEST_TIMEOUT", 15.0)),
    slow_threshold_ms=max(1.0, _env_float("AGENT_SLOW_THRESHOLD_MS", 1000.0)),
    llm_max_retries=max(1, _env_int("LLM_MAX_RETRIES", 3)),
    reports_db=os.getenv("REPORTS_DB", "reports.db"),
    generated_tests_dir=os.getenv("GENERATED_TESTS_DIR", "generated_tests"),
    mount_demo=_env_bool("MOUNT_DEMO", True),
    restrict_targets=_env_bool("AGENT_RESTRICT_TARGETS", False),
    allowed_target_hosts=_env_list("AGENT_ALLOWED_TARGET_HOSTS"),
)

