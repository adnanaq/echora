"""
Programmatic enrichment modules for deterministic operations.
Following async-first architecture and configuration-driven patterns from lessons learned.
"""

from .api_fetcher import ApiFetcher
from .config import EnrichmentConfig
from .enrichment_pipeline import EnrichmentPipeline
from .id_extractor import PlatformIDExtractor

__all__ = [
    "PlatformIDExtractor",
    "ApiFetcher",
    "EnrichmentPipeline",
    "EnrichmentConfig",
]
