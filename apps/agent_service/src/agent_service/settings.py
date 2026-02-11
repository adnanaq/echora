"""Environment-driven settings for the agent gRPC service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    """Parses a boolean environment variable.

    Args:
        name: Environment variable name.
        default: Default value used when variable is not set.

    Returns:
        Parsed boolean value.
    """
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AgentServiceSettings:
    """Configuration values required to run the agent gRPC service."""

    host: str = os.getenv("AGENT_SERVICE_HOST", "0.0.0.0")
    port: int = int(os.getenv("AGENT_SERVICE_PORT", "50051"))

    llm_enabled: bool = _get_bool("AGENT_LLM_ENABLED", True)
    openai_base_url: str = os.getenv(
        "AGENT_OPENAI_BASE_URL", "http://127.0.0.1:8000/codex/v1"
    )
    openai_api_key: str = os.getenv("AGENT_OPENAI_API_KEY", "sk-dummy")
    openai_model: str = os.getenv("AGENT_OPENAI_MODEL", "gpt-5")

    default_max_turns: int = int(os.getenv("AGENT_DEFAULT_MAX_TURNS", "4"))
    qdrant_limit: int = int(os.getenv("AGENT_QDRANT_LIMIT", "10"))
