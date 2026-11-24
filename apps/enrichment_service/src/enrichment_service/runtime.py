"""Build runtime dependencies for enrichment_service.

This module defines the runtime container and startup factory used by
enrichment_service. It resolves configurable defaults for pipeline input and
artifact output locations, and configures the cache the pipeline uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import EnrichmentConfig, Settings
from http_cache import HTTPCacheManager, configure_cache

from .seed_cleaning_runner import SeedCleaningRunner


@dataclass(slots=True)
class EnrichmentRuntime:
    """Runtime dependencies owned by enrichment_service."""

    default_file_path: str
    output_dir: str
    enrichment: EnrichmentConfig
    cache: HTTPCacheManager
    seed_cleaning_runner: SeedCleaningRunner | None = None


async def build_runtime(settings: Settings) -> EnrichmentRuntime:
    """Initialize runtime state for enrichment_service.

    Args:
        settings: Resolved application settings.

    Returns:
        Runtime values used by route handlers.
    """
    enrichment = settings.enrichment
    cache = configure_cache(settings.cache, settings.redis)
    agent_host = (
        "127.0.0.1"
        if settings.service.agent_service_host == "0.0.0.0"  # noqa: S104
        else settings.service.agent_service_host
    )
    agent_service_address = f"{agent_host}:{settings.service.agent_service_port}"
    return EnrichmentRuntime(
        default_file_path=settings.service.enrichment_default_file_path,
        output_dir=settings.service.enrichment_output_dir,
        enrichment=enrichment,
        cache=cache,
        seed_cleaning_runner=SeedCleaningRunner(
            enrichment=enrichment,
            cache=cache,
            agent_service_address=agent_service_address,
        ),
    )
