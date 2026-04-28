"""
Platform ID extractor for anime sources.
Programmatic extraction of platform IDs from URLs - no AI needed.
Performance: ~0.001 seconds vs 5+ seconds for AI extraction.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class PlatformIDExtractor:
    """
    Extracts platform IDs from anime source URLs using regex patterns.
    100% deterministic - same input always produces same output.
    """

    # Platforms where the full URL is the identifier (substring match, first wins)
    _URL_KEYS: dict[str, str] = {
        "mal_url": "myanimelist.net/anime/",
        "anime_planet_url": "anime-planet.com/anime/",
        "anilist_url": "anilist.co/anime/",
        "anisearch_url": "anisearch.com/anime/",
    }

    # Platforms where a captured group from the URL is the identifier
    PATTERNS = {
        "kitsu_id": r"kitsu\.(?:io|app)/anime/([^/\?]+)",
        "anidb_id": r"anidb\.(?:net/anime/|info/a)(\d+)",
        "notify_id": r"notify\.moe/anime/([^/\?]+)",
        "livechart_id": r"livechart\.me/anime/(\d+)",
    }

    def extract_all_ids(self, offline_data: dict[str, Any]) -> dict[str, str | None]:
        """
        Extract all platform IDs from sources array.

        Args:
            offline_data: Offline anime data with 'sources' array

        Returns:
            Dictionary mapping platform names to extracted IDs

        """
        sources = offline_data.get("sources", [])
        ids: dict[str, str | None] = {k: None for k in self._URL_KEYS}
        ids.update({k: None for k in self.PATTERNS})

        for source in sources:
            for key, substring in self._URL_KEYS.items():
                if ids[key] is None and substring in source:
                    ids[key] = source
            for platform, pattern in self.PATTERNS.items():
                if ids[platform] is None:
                    try:
                        m = re.search(pattern, source, re.IGNORECASE)
                        if m:
                            ids[platform] = m.group(1)
                    except Exception as e:
                        logger.warning(f"Error extracting ID from {source}: {e}")

        # Log extracted IDs for context (following context-rich error messages pattern)
        non_null_ids = {k: v for k, v in ids.items() if v}
        if non_null_ids:
            logger.debug(f"Extracted IDs: {non_null_ids}")
        else:
            logger.warning(f"No platform IDs found in sources: {sources}")

        return ids

    def validate_ids(self, ids: dict[str, str | None]) -> dict[str, str]:
        """
        Validate and clean extracted IDs.

        Args:
            ids: Dictionary of extracted IDs

        Returns:
            Dictionary of valid IDs only
        """
        valid_ids = {}

        for platform, id_value in ids.items():
            if not id_value:
                continue

            # Validate numeric IDs
            if platform in [
                "anidb_id",
                "livechart_id",
            ]:
                try:
                    # Ensure it's a valid integer
                    int(id_value)
                    valid_ids[platform] = id_value
                except ValueError:
                    logger.warning(f"Invalid {platform}: {id_value} is not numeric")
            else:
                # For slug-based platforms, just ensure it's not empty
                if id_value.strip():
                    valid_ids[platform] = id_value.strip()

        return valid_ids
