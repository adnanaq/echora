"""How enrichment caches provider responses and crawl results.

The Redis connection itself is ``RedisConfig``; this holds what is cached and
for how long.
"""

from typing import Literal

from pydantic import Field

from .env_settings import EnvSettings

# A lifetime for a source with no field of its own.
DEFAULT_TTL = 86400


class CacheConfig(EnvSettings):
    """What enrichment caches and for how long."""

    cache_enabled: bool = Field(
        default=True, description="Cache provider responses and crawl results"
    )
    storage_type: Literal["redis"] = Field(
        default="redis", description="Cache storage backend"
    )
    force_cache: bool = Field(
        default=True,
        description=(
            "Cache and serve responses whatever their Cache-Control or Expires "
            "headers say; some provider APIs send headers that forbid caching"
        ),
    )
    always_revalidate: bool = Field(
        default=False,
        description="Ask the server whether a cached response is still current on every hit",
    )
    max_cache_key_length: int = Field(
        default=200,
        gt=0,
        description=(
            "Longest cache key kept as is; longer keys are replaced by their "
            "SHA-256 (64 hex characters) to stay readable and cheap in memory"
        ),
    )

    # Sources that supply characters are kept for 7 days. A franchise is
    # enriched one title at a time, and roughly a third of its characters recur
    # across those titles, so a 24-hour window expired between runs and every
    # shared character was crawled again.
    ttl_mal: int = Field(default=604800, ge=0, description="MyAnimeList - 7 days")
    ttl_anilist: int = Field(default=604800, ge=0, description="AniList - 7 days")
    ttl_anidb: int = Field(default=604800, ge=0, description="AniDB - 7 days")
    ttl_kitsu: int = Field(default=604800, ge=0, description="Kitsu - 7 days")
    ttl_anime_planet: int = Field(
        default=604800, ge=0, description="Anime-Planet - 7 days"
    )
    ttl_anisearch: int = Field(default=604800, ge=0, description="AniSearch - 7 days")
    # Broadcast timings shift week to week and this source carries no character
    # data, so it gains nothing from a longer window.
    ttl_animeschedule: int = Field(
        default=86400, ge=0, description="AnimeSchedule - 24 hours"
    )
    # A Cloudflare clearance was still honoured after 214 minutes on AniDB; a
    # rejected one costs only a challenge, after which it is replaced.
    cloudflare_clearance_max_ttl: int = Field(
        default=86400,
        gt=0,
        description="Longest a stored Cloudflare clearance is kept - 24 hours",
    )

    def ttl_for(self, service: str) -> int:
        """Return how long a source's responses are cached.

        Args:
            service: The source, as named in the ``ttl_`` fields, such as
                ``"mal"`` or ``"anime_planet"``.

        Returns:
            The lifetime in seconds, or ``DEFAULT_TTL`` for a source with no
            field of its own.

        Examples:
            >>> CacheConfig().ttl_for("mal")
            604800
            >>> CacheConfig().ttl_for("animeschedule")
            86400
            >>> CacheConfig().ttl_for("unknown")
            86400
        """
        return getattr(self, f"ttl_{service}", DEFAULT_TTL)
