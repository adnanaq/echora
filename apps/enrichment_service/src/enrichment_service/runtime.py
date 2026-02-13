"""Build runtime dependencies for enrichment_service.

This module defines the runtime container and startup factory used by
enrichment_service. It resolves configurable defaults for pipeline input and
artifact output locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from common.config import Settings


@dataclass(slots=True)
class EnrichmentRuntime:
    """Runtime dependencies owned by enrichment_service."""

    default_file_path: str
    output_dir: str


async def build_runtime(settings: Settings) -> EnrichmentRuntime:
    """Initialize runtime state for enrichment_service.

    Args:
        settings: Resolved application settings.

    Returns:
        Runtime values used by route handlers.
    """
    del settings
    default_file_path = os.getenv(
        "ENRICHMENT_DEFAULT_FILE_PATH",
        "data/qdrant_storage/anime-offline-database.json",
    )
    output_dir = os.getenv("ENRICHMENT_OUTPUT_DIR", "assets/seed_data")
    return EnrichmentRuntime(
        default_file_path=default_file_path,
        output_dir=output_dir,
    )
