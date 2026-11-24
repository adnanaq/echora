"""Build runtime dependencies for agent_service."""

from dataclasses import dataclass
from pathlib import Path

from common.config import Settings


@dataclass(slots=True)
class AgentRuntime:
    """Shared configuration and directories owned by ``agent_service``."""

    agent_data_dir: Path
    typesafe_api_key: str | None


def build_runtime(settings: Settings) -> AgentRuntime:
    """Resolve runtime dependencies for all agents from application settings.

    Args:
        settings: Resolved application settings.

    Returns:
        ``AgentRuntime`` holding the agent data directory and TypeSafe API key.
    """
    typesafe_api_key = (
        settings.agent.typesafe_api_key.get_secret_value()
        if settings.agent.typesafe_api_key is not None
        else None
    )
    return AgentRuntime(
        agent_data_dir=settings.agent.agent_data_dir,
        typesafe_api_key=typesafe_api_key,
    )
