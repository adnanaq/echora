"""Enrichment pipeline configuration.

Field names are the environment variable names in lower case, so
``enrichment_batch_size`` is read from ``ENRICHMENT_BATCH_SIZE`` and
``anidb_client`` from ``ANIDB_CLIENT``.
"""

import logging

from pydantic import Field

from .env_settings import EnvSettings

logger = logging.getLogger(__name__)


class EnrichmentConfig(EnvSettings):
    """How the enrichment pipeline runs and talks to AniDB."""

    enrichment_max_concurrent_browsers: int = Field(
        default=4,
        ge=1,
        description=(
            "Maximum Chrome browsers open at once in this process. Each costs about "
            "200-400 MB and 14 processes; 4 lets one anime's four browser-based "
            "providers (MAL, AniSearch, Anime-Planet, AniDB) run side by side"
        ),
    )
    enrichment_batch_size: int = Field(
        default=10, ge=1, le=100, description="Anime processed at the same time"
    )
    enrichment_temp_dir: str = Field(
        default="temp", description="Folder for each anime's intermediate files"
    )
    enrichment_skip_failed_apis: bool = Field(
        default=True, description="Carry on when one provider fails"
    )
    enrichment_verbose_logging: bool = Field(
        default=False, description="Log the configuration and more detail per anime"
    )

    # AniDB is the only provider that needs registered credentials: it answers
    # an unrecognised client with `<error code="302">` rather than data.
    anidb_client: str = Field(
        default="animeenrichment", description="Registered AniDB client name"
    )
    anidb_clientver: str = Field(default="1.0", description="AniDB client version")
    anidb_protover: str = Field(default="1", description="AniDB API protocol version")
    anidb_min_request_interval: float = Field(
        default=2.0, gt=0, description="Shortest gap between AniDB requests, in seconds"
    )
    anidb_max_request_interval: float = Field(
        default=10.0, gt=0, description="Longest gap between AniDB requests, in seconds"
    )
    anidb_error_cooldown_base: float = Field(
        default=5.0, ge=0, description="Base seconds to wait after an AniDB error"
    )
    anidb_max_retries: int = Field(
        default=3, ge=0, description="Retries before giving up on an AniDB request"
    )
    anidb_ban_cooldown: float = Field(
        default=300.0,
        ge=0,
        description="Seconds to stop contacting AniDB after it returns a 555 ban",
    )

    weighted_score_baseline: float = Field(
        default=6.2,
        description=(
            "What a typical anime scores across our providers. Recompute every "
            "5,000 newly enriched anime and whenever the provider set changes; "
            "see docs/merge_rules.md"
        ),
    )
    weighted_score_baseline_votes: int = Field(
        default=1000,
        ge=0,
        description=(
            "Baseline votes every anime carries, deciding how many real voters "
            "it takes before its own score wins; see docs/merge_rules.md"
        ),
    )

    def log_configuration(self) -> None:
        """Log the settings that shape a pipeline run."""
        logger.info(
            f"Enrichment pipeline: {self.enrichment_max_concurrent_browsers} browsers, "
            f"batches of {self.enrichment_batch_size}, "
            f"failed providers {'skipped' if self.enrichment_skip_failed_apis else 'fatal'}"
        )
