"""
Platform ID extractor for anime sources.
Programmatic extraction of platform IDs from URLs - no AI needed.
Performance: ~0.001 seconds vs 5+ seconds for AI extraction.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlatformIDExtractor:
    """
    Extracts platform IDs from anime source URLs using regex patterns.
    100% deterministic - same input always produces same output.
    """

    # Define regex patterns for each platform
    PATTERNS = {
        "mal_id": r"myanimelist\.net/anime/(\d+)",
        "anilist_id": r"anilist\.co/anime/(\d+)",
        "kitsu_id": r"kitsu\.(?:io|app)/anime/([^/\?]+)",
        "anidb_id": r"anidb\.(?:net/anime/|info/a)(\d+)",
        "anime_planet_slug": r"anime-planet\.com/anime/([^/\?]+)",
        "anisearch_id": r"anisearch\.com/anime/(?:index/)?(\d+)",
        "notify_id": r"notify\.moe/anime/([^/\?]+)",
        "livechart_id": r"livechart\.me/anime/(\d+)",
    }

    def extract_all_ids(self, offline_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Extract all platform IDs from sources array.

        Args:
            offline_data: Offline anime data with 'sources' array

        Returns:
            Dictionary mapping platform names to extracted IDs

        Performance: ~0.001 seconds for all platforms combined
        """
        sources = offline_data.get("sources", [])
        ids = {}

        # Extract IDs for each platform
        for platform, pattern in self.PATTERNS.items():
            ids[platform] = self._extract_id(sources, pattern)

        # Log extracted IDs for context (following context-rich error messages pattern)
        non_null_ids = {k: v for k, v in ids.items() if v}
        if non_null_ids:
            logger.debug(f"Extracted IDs: {non_null_ids}")
        else:
            logger.warning(f"No platform IDs found in sources: {sources}")

        return ids

    def _extract_id(self, sources: List[str], pattern: str) -> Optional[str]:
        """
        Extract single ID using regex pattern.

        Args:
            sources: List of source URLs
            pattern: Regex pattern to match

        Returns:
            Extracted ID or None if not found
        """
        for source in sources:
            try:
                match = re.search(pattern, source, re.IGNORECASE)
                if match:
                    return match.group(1)
            except Exception as e:
                logger.warning(f"Error extracting ID from {source}: {e}")

        return None

    def get_platform_from_url(self, url: str) -> Optional[str]:
        """
        Identify which platform a URL belongs to.

        Args:
            url: Source URL

        Returns:
            Platform name or None if not recognized
        """
        for platform, pattern in self.PATTERNS.items():
            if re.search(pattern, url, re.IGNORECASE):
                return platform.replace("_id", "").replace("_slug", "")
        return None

    def extract_animeschedule_search_terms(
        self, offline_data: Dict[str, Any]
    ) -> List[str]:
        """
        Extract potential search terms for AnimSchedule.
        AnimSchedule doesn't use IDs, so we need search terms.

        Args:
            offline_data: Offline anime data

        Returns:
            List of search terms to try
        """
        search_terms = []

        # Primary title
        if title := offline_data.get("title"):
            search_terms.append(title)

        # English title
        if title_en := offline_data.get("title_english"):
            if title_en not in search_terms:
                search_terms.append(title_en)

        # Japanese title
        if title_jp := offline_data.get("title_japanese"):
            if title_jp not in search_terms:
                search_terms.append(title_jp)

        # Synonyms
        for synonym in offline_data.get("synonyms", []):
            if synonym and synonym not in search_terms:
                search_terms.append(synonym)

        # Limit to first 3 unique terms for efficiency
        return search_terms[:3]

    def validate_ids(self, ids: Dict[str, Optional[str]]) -> Dict[str, str]:
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
                "mal_id",
                "anilist_id",
                "anidb_id",
                "anisearch_id",
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
