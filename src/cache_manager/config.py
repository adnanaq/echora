"""
Cache configuration for HTTP requests in enrichment pipeline.

Supports Redis backend for production and multi-agent concurrent processing.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """HTTP cache configuration for enrichment pipeline."""

    enabled: bool = Field(
        default=True,
        description="Enable HTTP caching (enabled by default on feature branch)",
    )

    storage_type: Literal["redis"] = Field(
        default="redis", description="Cache storage backend type"
    )

    # Redis configuration
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # Service-specific TTLs (in seconds) - all set to 24 hours for consistency
    ttl_jikan: int = Field(
        default=86400, description="Jikan (MyAnimeList) cache TTL - 24 hours"
    )
    ttl_anilist: int = Field(default=86400, description="AniList cache TTL - 24 hours")
    ttl_anidb: int = Field(default=86400, description="AniDB cache TTL - 24 hours")
    ttl_kitsu: int = Field(default=86400, description="Kitsu cache TTL - 24 hours")
    ttl_anime_planet: int = Field(
        default=86400, description="Anime-Planet cache TTL - 24 hours"
    )
    ttl_anisearch: int = Field(
        default=86400, description="AniSearch cache TTL - 24 hours"
    )
    ttl_animeschedule: int = Field(
        default=86400, description="AnimSchedule cache TTL - 24 hours"
    )

    # Redis connection pool configuration
    redis_max_connections: int = Field(
        default=100,
        description="Max Redis connections (tuned for multi-agent concurrency: 20 agents × 10 concurrent ops)",
    )
    redis_socket_keepalive: bool = Field(
        default=True,
        description="Enable TCP keepalive to detect stale connections and prevent NAT/firewall timeouts",
    )
    redis_socket_connect_timeout: int = Field(
        default=5,
        description="Connection timeout in seconds (fail-fast on unreachable Redis)",
    )
    redis_socket_timeout: int = Field(
        default=10,
        description="Socket read/write timeout in seconds (prevents hanging on slow Redis operations)",
    )
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Retry operations on timeout (safe for idempotent cache operations)",
    )
    redis_health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds (0=disabled, proactively validates connections)",
    )

    class Config:
        """Pydantic configuration."""

        env_prefix = ""  # Allow both prefixed and non-prefixed env vars


def get_cache_config() -> CacheConfig:
    """
    Get cache configuration from environment variables.

    Environment Variables:
        ENABLE_HTTP_CACHE: Enable caching (default: true)
        REDIS_CACHE_URL: Redis connection URL (default: redis://localhost:6379/0)
        REDIS_MAX_CONNECTIONS: Max connection pool size (default: 100)
        REDIS_SOCKET_KEEPALIVE: Enable TCP keepalive (default: true)
        REDIS_SOCKET_CONNECT_TIMEOUT: Connection timeout in seconds (default: 5)
        REDIS_SOCKET_TIMEOUT: Read/write timeout in seconds (default: 10)
        REDIS_RETRY_ON_TIMEOUT: Retry on timeout (default: true)
        REDIS_HEALTH_CHECK_INTERVAL: Health check interval in seconds (default: 30)

    Returns:
        CacheConfig instance
    """
    import os

    return CacheConfig(
        enabled=os.getenv("ENABLE_HTTP_CACHE", "true").lower() == "true",
        redis_url=os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/0"),
    )
