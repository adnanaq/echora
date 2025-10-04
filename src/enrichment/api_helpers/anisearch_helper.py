"""AniSearch enrichment helper wrapping the AniSearch scraper."""

import logging
from typing import Any, Dict, Optional

from ..scrapers.anisearch_scraper import AniSearchScraper

logger = logging.getLogger(__name__)


class AniSearchEnrichmentHelper:
    """Helper class for fetching AniSearch data using the scraper."""

    def __init__(self):
        """Initialize AniSearch helper."""
        self.scraper = AniSearchScraper()

    async def fetch_all_data(self, anisearch_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive anime data from AniSearch.

        Args:
            anisearch_id: AniSearch anime ID (e.g., 18878 for Dandadan)

        Returns:
            Dictionary with anime data or None if fetch fails
        """
        try:
            logger.info(f"Fetching AniSearch data for ID {anisearch_id}")

            # Use scraper to fetch anime data by ID
            result = await self.scraper.get_anime_by_id(anisearch_id)

            if result:
                logger.info(
                    f"Successfully fetched AniSearch data for ID {anisearch_id}: {result.get('title', 'Unknown')}"
                )
                return result
            else:
                logger.warning(
                    f"AniSearch returned no data for ID {anisearch_id}"
                )
                return None

        except Exception as e:
            logger.error(f"Error fetching AniSearch data for ID {anisearch_id}: {e}")
            return None

    async def close(self) -> None:
        """Close the scraper session."""
        if self.scraper:
            await self.scraper.close()
