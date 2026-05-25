"""Redis configuration for shared Redis connectivity."""

from pydantic import BaseModel, Field


class RedisConfig(BaseModel):
    """Configuration for Redis connection used by embedding cache and other services."""

    redis_url: str | None = Field(
        default=None, description="Redis connection URL (e.g. redis://localhost:6379/0)"
    )
    redis_max_connections: int = Field(
        default=20, ge=1, description="Maximum number of connections in the Redis pool"
    )
    redis_socket_connect_timeout: int = Field(
        default=5, gt=0, description="Socket connect timeout in seconds"
    )
    redis_socket_timeout: int = Field(
        default=10, gt=0, description="Socket read/write timeout in seconds"
    )
