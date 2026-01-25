"""
Configuration for programmatic enrichment pipeline.
Following configuration-driven patterns from lessons learned.
"""

import logging

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class EnrichmentConfig(BaseSettings):
    """
    Enrichment pipeline configuration with validation.
    Follows the configuration-first approach from lessons learned.
    """

    # API Configuration
    api_timeout: int = Field(
        default=200,
        description="Timeout for each API call in seconds (200s allows ~400 detailed Jikan requests at 0.5s each)",
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
        default="data/anime-offline-database.json",
        description="Path to offline anime database",
    )
    enriched_database_path: str = Field(
        default="assets/seed_data/anime_database.json",
        description="Path to enriched anime database",
    )
    temp_dir: str = Field(
        default="temp", description="Temporary directory for processing"
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
        Validate that an API timeout is between 1 and 300 seconds.

        Parameters:
            v (int): The timeout value in seconds to validate.

        Returns:
            int: The validated timeout value.

        Raises:
            ValueError: If `v` is less than 1 or greater than 300.
        """
        if v < 1 or v > 300:
            raise ValueError("API timeout must be between 1 and 300 seconds")
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

    class Config:
        env_prefix = "ENRICHMENT_"
        case_sensitive = False

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
