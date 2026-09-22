"""
Configuration for programmatic enrichment pipeline.
Following configuration-driven patterns from lessons learned.
"""

import logging

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class EnrichmentConfig(BaseSettings):
    """
    Enrichment pipeline configuration with validation.
    Follows the configuration-first approach from lessons learned.
    """

    # API Configuration
    api_timeout: int = Field(
        default=200,
        description="Timeout for each API call in seconds (200s allows ~400 detailed MAL requests at 0.5s each)",
    )
    max_concurrent_apis: int = Field(
        default=6, description="Maximum concurrent API calls"
    )
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed API calls"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )

    # Batch Processing
    batch_size: int = Field(
        default=10, description="Number of anime to process concurrently"
    )
    episode_batch_size: int = Field(
        default=50, description="Episodes to process per batch"
    )
    character_batch_size: int = Field(
        default=50, description="Characters to process per batch"
    )

    # Performance Tuning
    enable_caching: bool = Field(
        default=True, description="Enable API response caching"
    )
    cache_ttl: int = Field(default=86400, description="Cache TTL in seconds (24 hours)")
    connection_pool_size: int = Field(
        default=100, description="Total connection pool size"
    )
    connections_per_host: int = Field(
        default=10, description="Connections per host limit"
    )

    # Data Paths
    offline_database_path: str = Field(
        default="assets/seed_data/anime-offline-database.json",
        description="Path to offline anime database",
    )
    enriched_database_path: str = Field(
        default="assets/seed_data/anime_database.json",
        description="Path to enriched anime database",
    )
    temp_dir: str = Field(
        default="temp", description="Temporary directory for processing"
    )

    # AniDB
    # AniDB is the only provider that needs registered credentials: it answers
    # an unrecognised client with `<error code="302">` rather than data. These
    # keep their bare ANIDB_ names via validation_alias, so the ENRICHMENT_
    # prefix does not apply and the names already in .env keep working.
    anidb_client: str = Field(
        default="animeenrichment",
        validation_alias="ANIDB_CLIENT",
        description="Registered AniDB client name",
    )
    anidb_clientver: str = Field(
        default="1.0",
        validation_alias="ANIDB_CLIENTVER",
        description="AniDB client version",
    )
    anidb_protover: str = Field(
        default="1",
        validation_alias="ANIDB_PROTOVER",
        description="AniDB API protocol version",
    )
    anidb_min_request_interval: float = Field(
        default=2.0,
        validation_alias="ANIDB_MIN_REQUEST_INTERVAL",
        description="Shortest gap between AniDB requests, in seconds",
    )
    anidb_max_request_interval: float = Field(
        default=10.0,
        validation_alias="ANIDB_MAX_REQUEST_INTERVAL",
        description="Longest gap between AniDB requests, in seconds",
    )
    anidb_error_cooldown_base: float = Field(
        default=5.0,
        validation_alias="ANIDB_ERROR_COOLDOWN_BASE",
        description="Base seconds to wait after an AniDB error response",
    )
    anidb_max_retries: int = Field(
        default=3,
        validation_alias="ANIDB_MAX_RETRIES",
        description="Retries before giving up on an AniDB request",
    )
    anidb_circuit_breaker_threshold: int = Field(
        default=5,
        validation_alias="ANIDB_CIRCUIT_BREAKER_THRESHOLD",
        description="Consecutive AniDB failures that open the circuit",
    )
    anidb_circuit_breaker_timeout: float = Field(
        default=300.0,
        validation_alias="ANIDB_CIRCUIT_BREAKER_TIMEOUT",
        description="Seconds the AniDB circuit stays open",
    )

    # Feature Flags
    skip_failed_apis: bool = Field(
        default=True,
        description="Continue processing if an API fails (graceful degradation)",
    )
    no_timeout_mode: bool = Field(
        default=False,
        description="Disable timeouts for background processing (fetch ALL data)",
    )
    validate_schemas: bool = Field(
        default=True, description="Validate output against AnimeRecord schema"
    )
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")

    @field_validator("api_timeout")
    def validate_timeout(cls, v):
        """
        Validate that an API timeout is between 1 and 3600 seconds.

        Parameters:
            v (int): The timeout value in seconds to validate.

        Returns:
            int: The validated timeout value.

        Raises:
            ValueError: If `v` is less than 1 or greater than 3600.
        """
        if v < 1 or v > 3600:
            raise ValueError("API timeout must be between 1 and 3600 seconds")
        return v

    @field_validator("batch_size")
    def validate_batch_size(cls, v):
        if v < 1 or v > 100:
            raise ValueError("Batch size must be between 1 and 100")
        return v

    @field_validator("cache_ttl")
    def validate_cache_ttl(cls, v):
        if v < 0:
            raise ValueError("Cache TTL must be non-negative")
        return v

    model_config = SettingsConfigDict(
        env_prefix="ENRICHMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def log_configuration(self) -> None:
        """Log current configuration for debugging (context-rich errors)."""
        logger.info("Enrichment Pipeline Configuration:")
        logger.info(f"  API Timeout: {self.api_timeout}s")
        logger.info(f"  Max Concurrent APIs: {self.max_concurrent_apis}")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Caching: {'Enabled' if self.enable_caching else 'Disabled'}")
        logger.info(
            f"  Graceful Degradation: {'Enabled' if self.skip_failed_apis else 'Disabled'}"
        )
