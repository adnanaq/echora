"""AniSearch enrichment helper using crawlers (anime, episodes, characters)."""

import logging
import os
from typing import Any

from common.utils.jsonl_utils import append_jsonl

from ..crawlers.anisearch.anisearch_anime_crawler import (
    BASE_ANIME_URL,
    fetch_anisearch_anime,
)
from ..crawlers.anisearch_character_crawler import fetch_anisearch_characters
from ..crawlers.anisearch_episode_crawler import fetch_anisearch_episodes
from .base_helper import BaseEnrichmentHelper

logger = logging.getLogger(__name__)


class AniSearchEnrichmentHelper(BaseEnrichmentHelper):
    """Helper class for fetching AniSearch data using crawlers."""

    def __init__(self) -> None:
        """Initialize AniSearch helper (crawlers are stateless)."""

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetches anime data from AniSearch and enriches it with optional episodes and characters.

        Args:
            ids: Dictionary of validated platform IDs/URLs. Must contain 'anisearch_id'.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            The anime data dictionary enriched with optional "episodes" and "characters" keys, or `None` if the primary anime data could not be fetched.
        """
        anisearch_id = ids.get("anisearch_id")
        if not anisearch_id:
            return None

        output_path = os.path.join(temp_dir, "anisearch.jsonl") if temp_dir else None

        return await self._fetch_all_impl(
            int(anisearch_id),
            include_episodes=True,
            include_characters=True,
            output_path=output_path,
        )

    async def _fetch_all_impl(
        self,
        anisearch_id: int,
        include_episodes: bool = True,
        include_characters: bool = True,
        output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Internal implementation of fetch_all.

        Parameters:
            anisearch_id (int): AniSearch numeric anime ID (e.g., 18878 for Dandadan).
            include_episodes (bool): If True, attempt to attach an "episodes" list to the returned data.
            include_characters (bool): If True, attempt to attach a "characters" list to the returned data.
            output_path (str): Optional path to write JSONL output.

        Returns:
            The anime data dictionary enriched with optional "episodes" and "characters" keys, or `None` if the primary anime data could not be fetched.
        """
        try:
            logger.info(f"Fetching all AniSearch data for ID {anisearch_id}")

            # Fetch anime data (primary data)
            anime_data = await self.fetch_anime(anisearch_id)

            if not anime_data:
                logger.warning(f"Failed to fetch anime data for ID {anisearch_id}")
                return None

            if output_path:
                append_jsonl(output_path, anime_data)

            # Fetch episodes if requested
            if include_episodes:
                try:
                    episode_data = await self.fetch_episodes(anisearch_id)
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
                    character_data = await self.fetch_characters(anisearch_id)
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
            logger.exception(f"Error in _fetch_all_impl for ID {anisearch_id}")
            return None

    async def fetch_anime(self, anisearch_id: int) -> dict[str, Any] | None:
        """
        Fetches anime metadata from AniSearch for the given AniSearch numeric ID.

        Parameters:
            anisearch_id (int): AniSearch anime ID (for example, 18878 for Dandadan).

        Returns:
            dict: Anime data dictionary if available, `None` otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch anime data for ID {anisearch_id}")

            anime_data = await fetch_anisearch_anime(
                f"{BASE_ANIME_URL}{anisearch_id}",
                output_path=None,
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

    async def fetch_episodes(
        self, anisearch_id: int
    ) -> list[dict[str, Any]] | None:
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

            logger.info(
                f"Successfully fetched {len(episode_data)} episodes for ID {anisearch_id}"
            )
            return episode_data

        except Exception:
            logger.exception(f"Error fetching episode data for ID {anisearch_id}")
            return None

    async def fetch_characters(self, anisearch_id: int) -> dict[str, Any] | None:
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

            char_count = len(character_data.get("characters", []))
            logger.info(
                f"Successfully fetched {char_count} characters for ID {anisearch_id}"
            )
            return character_data

        except Exception:
            logger.exception(f"Error fetching character data for ID {anisearch_id}")
            return None

