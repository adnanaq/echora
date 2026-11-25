"""Build runtime dependencies for enrichment_service.

This module defines the runtime container and startup factory used by
enrichment_service. It resolves configurable defaults for pipeline input and
artifact output locations, and configures the cache the pipeline uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from common.config import EnrichmentConfig, Settings
from http_cache import HTTPCacheManager, configure_cache

if TYPE_CHECKING:
    from enrichment.sources.base.page_fetcher import PageFetcher


@dataclass(slots=True)
class EnrichmentRuntime:
    """Runtime dependencies owned by enrichment_service."""

    default_file_path: str
    output_dir: str
    enrichment: EnrichmentConfig
    cache: HTTPCacheManager
    page_fetchers: dict[str, PageFetcher] = field(default_factory=dict)


async def build_runtime(settings: Settings) -> EnrichmentRuntime:
    """Initialize runtime state for enrichment_service.

    Args:
        settings: Resolved application settings.

    Returns:
        Runtime values used by route handlers.
    """
    from .routes.fetch_page import build_page_fetchers

    runtime = EnrichmentRuntime(
        default_file_path=settings.service.enrichment_default_file_path,
        output_dir=settings.service.enrichment_output_dir,
        enrichment=settings.enrichment,
        cache=configure_cache(settings.cache, settings.redis),
    )
    runtime.page_fetchers = build_page_fetchers(runtime)
    return runtime
