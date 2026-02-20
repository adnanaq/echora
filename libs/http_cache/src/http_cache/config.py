"""
Cache configuration for HTTP requests in enrichment pipeline.

Supports Redis backend for production and multi-agent concurrent processing.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheConfig(BaseSettings):
    """HTTP cache configuration for enrichment pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    cache_enabled: bool = Field(
        default=True,
        description="Enable HTTP caching (enabled by default on feature branch)",
    )

    storage_type: Literal["redis"] = Field(
        default="redis", description="Cache storage backend type"
    )

    # Redis configuration
    redis_url: str | None = Field(
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

    # Cache behavior configuration
    force_cache: bool = Field(
        default=True,
        description=(
            "If True, ignores RFC 9111 headers and always caches/serves from cache "
            "regardless of Cache-Control, Expires, etc. Useful for misbehaving APIs."
        ),
    )
    always_revalidate: bool = Field(
        default=False,
        description="If True, always attempts to revalidate with the server even on cache hits.",
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


@lru_cache
def get_cache_config() -> CacheConfig:
    """Get cached CacheConfig instance populated from environment variables.

    Environment variables are automatically read by Pydantic BaseSettings:
        CACHE_ENABLED (default: true)
        FORCE_CACHE (default: true)
        ALWAYS_REVALIDATE (default: false)
        MAX_CACHE_KEY_LENGTH (default: 200)
        REDIS_URL (default: "redis://localhost:6379/0")
        REDIS_MAX_CONNECTIONS (default: 100)
        REDIS_SOCKET_KEEPALIVE (default: true)
        REDIS_SOCKET_CONNECT_TIMEOUT (default: 5)
        REDIS_SOCKET_TIMEOUT (default: 10)
        REDIS_RETRY_ON_TIMEOUT (default: true)
        REDIS_HEALTH_CHECK_INTERVAL (default: 30)
        TTL_JIKAN, TTL_ANILIST, TTL_ANIDB, etc. (default: 86400)

    Returns:
        Cached CacheConfig instance.

    Note:
        Uses @lru_cache for singleton pattern. For testing, call
        get_cache_config.cache_clear() to reset the cache.
    """
    return CacheConfig()
