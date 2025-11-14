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

    class Config:
        """Pydantic configuration."""

        env_prefix = ""  # Allow both prefixed and non-prefixed env vars


def get_cache_config() -> CacheConfig:
    """
    Get cache configuration from environment variables.

    Environment Variables:
        ENABLE_HTTP_CACHE: Enable caching (default: true)
        REDIS_CACHE_URL: Redis connection URL (default: redis://localhost:6379/0)

    Returns:
        CacheConfig instance
    """
    import os

    return CacheConfig(
        enabled=os.getenv("ENABLE_HTTP_CACHE", "true").lower() == "true",
        redis_url=os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/0"),
    )
