"""AniSearch enrichment helper using crawlers (anime, episodes, characters)."""

import logging
import re
from typing import Any, Dict, List, Optional

from ..crawlers.anisearch_anime_crawler import fetch_anisearch_anime
from ..crawlers.anisearch_character_crawler import fetch_anisearch_characters
from ..crawlers.anisearch_episode_crawler import fetch_anisearch_episodes

logger = logging.getLogger(__name__)


class AniSearchEnrichmentHelper:
    """Helper class for fetching AniSearch data using crawlers."""

    def __init__(self) -> None:
        """Initialize AniSearch helper (crawlers are stateless)."""

    async def extract_anisearch_id_from_url(self, url: str) -> Optional[int]:
        """Extract AniSearch ID from URL.

        Args:
            url: AniSearch URL (e.g., "https://www.anisearch.com/anime/18878,dan-da-dan")

        Returns:
            AniSearch ID (e.g., 18878) or None if extraction fails
        """
        try:
            # Pattern: https://www.anisearch.com/anime/18878,dan-da-dan
            # or: https://www.anisearch.com/anime/18878
            match = re.search(r"anisearch\.com/anime/(\d+)", url)
            if match:
                return int(match.group(1))
            return None
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error extracting AniSearch ID from URL {url}: {e}")
            return None

    async def find_anisearch_url(
        self, offline_anime_data: Dict[str, Any]
    ) -> Optional[str]:
        """Find AniSearch URL from offline anime data sources.

        Args:
            offline_anime_data: Offline anime data containing sources list

        Returns:
            AniSearch URL or None if not found
        """
        try:
            sources = offline_anime_data.get("sources", [])
            for source in sources:
                if isinstance(source, str) and "anisearch.com" in source:
                    return source
            return None
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"Error finding AniSearch URL: {e}")
            return None

    async def fetch_anime_data(self, anisearch_id: int) -> Optional[Dict[str, Any]]:
        """Fetch anime data using anisearch_anime_crawler.

        Args:
            anisearch_id: AniSearch anime ID (e.g., 18878 for Dandadan)

        Returns:
            Dict containing anime data or None if fetch fails
        """
        try:
            # Construct URL from ID
            url = f"https://www.anisearch.com/anime/{anisearch_id}"

            logger.info(f"Fetching AniSearch anime data for ID {anisearch_id}")

            # Call anime crawler
            anime_data = await fetch_anisearch_anime(
                url=url,
                return_data=True,
                output_path=None  # No file output - return data only
            )

            if not anime_data:
                logger.warning(f"Anime crawler returned no data for ID {anisearch_id}")
                return None

            logger.info(
                f"Successfully fetched anime data for ID {anisearch_id}: "
                f"{anime_data.get('japanese_title', 'Unknown')}"
            )
            return anime_data

        except Exception as e:
            logger.error(f"Error fetching anime data for ID {anisearch_id}: {e}")
            return None

    async def fetch_episode_data(self, anisearch_id: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch episode data using anisearch_episode_crawler.

        Args:
            anisearch_id: AniSearch anime ID

        Returns:
            List of episode dicts or None if fetch fails
        """
        try:
            # Construct episodes URL
            url = f"https://www.anisearch.com/anime/{anisearch_id}/episodes"

            logger.info(f"Fetching AniSearch episode data for ID {anisearch_id}")

            # Call episode crawler
            episode_data = await fetch_anisearch_episodes(
                url=url,
                return_data=True,
                output_path=None  # No file output - return data only
            )

            if not episode_data:
                logger.debug(f"No episode data found for ID {anisearch_id}")
                return None

            logger.info(f"Successfully fetched {len(episode_data)} episodes for ID {anisearch_id}")
            return episode_data

        except Exception as e:
            logger.error(f"Error fetching episode data for ID {anisearch_id}: {e}")
            return None

    async def fetch_character_data(self, anisearch_id: int) -> Optional[Dict[str, Any]]:
        """Fetch character data using anisearch_character_crawler.

        Args:
            anisearch_id: AniSearch anime ID

        Returns:
            Dict containing character data or None if fetch fails
        """
        try:
            # Construct characters URL
            url = f"https://www.anisearch.com/anime/{anisearch_id}/characters"

            logger.info(f"Fetching AniSearch character data for ID {anisearch_id}")

            # Call character crawler
            character_data = await fetch_anisearch_characters(
                url=url,
                return_data=True,
                output_path=None  # No file output - return data only
            )

            if not character_data:
                logger.debug(f"No character data found for ID {anisearch_id}")
                return None

            char_count = character_data.get("total_count", 0)
            logger.info(f"Successfully fetched {char_count} characters for ID {anisearch_id}")
            return character_data

        except Exception as e:
            logger.error(f"Error fetching character data for ID {anisearch_id}: {e}")
            return None

    async def fetch_all_data(
        self,
        anisearch_id: int,
        include_episodes: bool = True,
        include_characters: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive AniSearch data (anime + episodes + characters).

        Args:
            anisearch_id: AniSearch anime ID (e.g., 18878 for Dandadan)
            include_episodes: Whether to fetch episode data (default: True)
            include_characters: Whether to fetch character data (default: True)

        Returns:
            Dict containing complete AniSearch data or None if fetch fails
        """
        try:
            logger.info(f"Fetching all AniSearch data for ID {anisearch_id}")

            # Fetch anime data (primary data)
            anime_data = await self.fetch_anime_data(anisearch_id)

            if not anime_data:
                logger.warning(f"Failed to fetch anime data for ID {anisearch_id}")
                return None

            # Fetch episodes if requested
            if include_episodes:
                try:
                    episode_data = await self.fetch_episode_data(anisearch_id)
                    if episode_data:
                        anime_data["episodes"] = episode_data
                        logger.info(f"Integrated {len(episode_data)} episodes into anime data")
                except Exception as e:
                    logger.warning(f"Failed to fetch episodes for ID {anisearch_id}: {e}")
                    # Continue without episodes - non-critical

            # Fetch characters if requested
            if include_characters:
                try:
                    character_data = await self.fetch_character_data(anisearch_id)
                    if character_data:
                        anime_data["characters"] = character_data.get("characters", [])
                        logger.info(
                            f"Integrated {character_data.get('total_count', 0)} characters into anime data"
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch characters for ID {anisearch_id}: {e}")
                    # Continue without characters - non-critical

            logger.info(f"Successfully fetched all AniSearch data for ID {anisearch_id}")
            return anime_data

        except Exception as e:
            logger.error(f"Error in fetch_all_data for ID {anisearch_id}: {e}")
            return None

    async def close(self) -> None:
        """Cleanup (crawlers are stateless, no cleanup needed)."""
