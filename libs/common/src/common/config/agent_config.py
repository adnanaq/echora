"""Agent service configuration."""

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for the internal agent gRPC service."""

    service_host: str = Field(
        default="0.0.0.0",
        description="Bind host for the internal agent gRPC service",
    )
    service_port: int = Field(
        default=50051,
        description="Bind port for the internal agent gRPC service",
    )

    llm_enabled: bool = Field(
        default=True,
        description="Enable LLM-driven orchestration in SearchAI",
    )
    openai_base_url: str = Field(
        default="http://127.0.0.1:8000/codex/v1",
        description="OpenAI-compatible base URL used by the agent runtime",
    )
    openai_api_key: str = Field(
        default="sk-dummy",
        description="API key used by the OpenAI-compatible client",
    )
    openai_model: str = Field(
        default="gpt-5",
        description="Model name used by planner/sufficiency agents",
    )

    default_max_turns: int = Field(
        default=4,
        ge=1,
        description="Default planner loop cap when request max_turns is not set",
    )
    qdrant_limit: int = Field(
        default=10,
        ge=1,
        description="Max Qdrant hits retrieved per retrieval step",
    )
