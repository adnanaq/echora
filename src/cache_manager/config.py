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

    # Cache key configuration
    max_cache_key_length: int = Field(
        default=200,
        description=(
            "Maximum cache key length before hashing. "
            "Redis has a 512MB key limit, but shorter keys provide: "
            "1. Better readability in Redis CLI/monitoring tools "
            "2. Improved lookup performance "
            "3. Memory efficiency (keys stored in memory). "
            "Keys exceeding this threshold are SHA256-hashed to 64 hex chars."
        ),
    )

    # Redis connection pool configuration
    redis_max_connections: int = Field(
        default=100,
        description="Max Redis connections (tuned for multi-agent concurrency: 20 agents x 10 concurrent ops)",
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

    def parse_bool(value: str, default: bool) -> bool:
        """Parse boolean from environment variable string."""
        if value.lower() in ("true", "1", "yes"):
            return True
        elif value.lower() in ("false", "0", "no"):
            return False
        return default

    def parse_int(value: str, default: int) -> int:
        """Parse integer from environment variable string."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    return CacheConfig(
        enabled=os.getenv("ENABLE_HTTP_CACHE", "true").lower() == "true",
        redis_url=os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/0"),
        redis_max_connections=parse_int(
            os.getenv("REDIS_MAX_CONNECTIONS", "100"), 100
        ),
        redis_socket_keepalive=parse_bool(
            os.getenv("REDIS_SOCKET_KEEPALIVE", "true"), True
        ),
        redis_socket_connect_timeout=parse_int(
            os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"), 5
        ),
        redis_socket_timeout=parse_int(
            os.getenv("REDIS_SOCKET_TIMEOUT", "10"), 10
        ),
        redis_retry_on_timeout=parse_bool(
            os.getenv("REDIS_RETRY_ON_TIMEOUT", "true"), True
        ),
        redis_health_check_interval=parse_int(
            os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"), 30
        ),
    )
