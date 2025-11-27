"""AniSearch enrichment helper using crawlers (anime, episodes, characters)."""

import logging
import re
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

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
        """
        Fetches anime metadata from AniSearch for the given AniSearch numeric ID.
        
        Parameters:
            anisearch_id (int): AniSearch anime ID (for example, 18878 for Dandadan).
        
        Returns:
            dict: Anime data dictionary if available, `None` otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch anime data for ID {anisearch_id}")

            # Call anime crawler with anime_id (accepts ID, path, or full URL)
            anime_data = await fetch_anisearch_anime(
                anime_id=str(anisearch_id),  # Pass ID directly
                output_path=None,  # No file output - return data only
            )

            if not anime_data:
                logger.warning(f"Anime crawler returned no data for ID {anisearch_id}")
                return None

            logger.info(
                f"Successfully fetched anime data for ID {anisearch_id}: "
                f"{anime_data.get('japanese_title', 'Unknown')}"
            )
            return anime_data

        except Exception:
            logger.exception(f"Error fetching anime data for ID {anisearch_id}")
            return None

    async def fetch_episode_data(
        self, anisearch_id: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch episode data for a given AniSearch anime ID.
        
        Parameters:
            anisearch_id (int): AniSearch numeric anime ID.
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of episode dictionaries if found, `None` otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch episode data for ID {anisearch_id}")

            # Call episode crawler with anime_id (accepts ID, path, or full URL)
            episode_data = await fetch_anisearch_episodes(
                anime_id=str(anisearch_id),  # Pass ID directly
                output_path=None,  # No file output - return data only
            )

            if not episode_data:
                logger.debug(f"No episode data found for ID {anisearch_id}")
                return None
            else:
                logger.info(
                    f"Successfully fetched {len(episode_data)} episodes for ID {anisearch_id}"
                )
                return episode_data

        except Exception:
            logger.exception(f"Error fetching episode data for ID {anisearch_id}")
            return None

    async def fetch_character_data(self, anisearch_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve character information for a given AniSearch anime ID.

        Parameters:
            anisearch_id (int): AniSearch numeric ID of the anime.

        Returns:
            dict: Character data dictionary with `characters` key if successful, `None` otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch character data for ID {anisearch_id}")

            # Call character crawler with anime_id (accepts ID, path, or full URL)
            character_data = await fetch_anisearch_characters(
                anime_id=str(anisearch_id),  # Pass ID directly
                output_path=None,  # No file output - return data only
            )

            if not character_data:
                logger.debug(f"No character data found for ID {anisearch_id}")
                return None
            else:
                char_count = len(character_data.get("characters", []))
                logger.info(
                    f"Successfully fetched {char_count} characters for ID {anisearch_id}"
                )
                return character_data

        except Exception:
            logger.exception(f"Error fetching character data for ID {anisearch_id}")
            return None

    async def fetch_all_data(
        self,
        anisearch_id: int,
        include_episodes: bool = True,
        include_characters: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches anime data from AniSearch and enriches it with optional episodes and characters.
        
        Parameters:
            anisearch_id (int): AniSearch numeric anime ID (e.g., 18878 for Dandadan).
            include_episodes (bool): If True, attempt to attach an "episodes" list to the returned data.
            include_characters (bool): If True, attempt to attach a "characters" list to the returned data.
        
        Returns:
            The anime data dictionary enriched with optional "episodes" and "characters" keys, or `None` if the primary anime data could not be fetched.
        """
        try:
            logger.info(f"Fetching all AniSearch data for ID {anisearch_id}")

            # Fetch anime data (primary data)
            anime_data = await self.fetch_anime_data(anisearch_id)

            if not anime_data:
                logger.warning(f"Failed to fetch anime data for ID {anisearch_id}")
                return None
            else:
                # Fetch episodes if requested
                if include_episodes:
                    try:
                        episode_data = await self.fetch_episode_data(anisearch_id)
                        if episode_data:
                            anime_data["episodes"] = episode_data
                            logger.info(
                                f"Integrated {len(episode_data)} episodes into anime data"
                            )
                    except Exception:
                        logger.warning(
                            f"Failed to fetch episodes for ID {anisearch_id}",
                            exc_info=True,
                        )
                        # Continue without episodes - non-critical

                # Fetch characters if requested
                if include_characters:
                    try:
                        character_data = await self.fetch_character_data(anisearch_id)
                        if character_data:
                            characters = character_data.get("characters", [])
                            anime_data["characters"] = characters
                            logger.info(
                                f"Integrated {len(characters)} characters into anime data"
                            )
                    except Exception:
                        logger.warning(
                            f"Failed to fetch characters for ID {anisearch_id}",
                            exc_info=True,
                        )
                        # Continue without characters - non-critical

                logger.info(
                    f"Successfully fetched all AniSearch data for ID {anisearch_id}"
                )
                return anime_data

        except Exception:
            logger.exception(f"Error in fetch_all_data for ID {anisearch_id}")
            return None

    async def close(self) -> None:
        """
        Perform cleanup for the helper; currently a no-op since crawlers are stateless.
        """

    async def __aenter__(self) -> "AniSearchEnrichmentHelper":
        """
        Enter the asynchronous context and make the helper available as the context value.
        
        Returns:
            AniSearchEnrichmentHelper: The same helper instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """
        Perform cleanup when exiting the async context.
        
        Calls the helper's cleanup routine and returns `False` to indicate any exception raised in the context should not be suppressed.
        
        Returns:
            `False` to indicate exceptions are not suppressed.
        """
        await self.close()
        return False