"""AniSearch enrichment helper using crawlers (anime, episodes, characters)."""

import logging
import os
from typing import Any

from common.utils.jsonl_utils import append_jsonl

from ..crawlers.anisearch.anisearch_anime_crawler import (
    BASE_ANIME_URL,
    fetch_anisearch_anime,
)
from ..crawlers.anisearch.anisearch_character_crawler import fetch_anisearch_characters
from ..crawlers.anisearch.anisearch_character_refs_crawler import (
    fetch_anisearch_character_refs,
)
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
        """Fetch all AniSearch data for this anime.

        Args:
            ids: Dictionary of validated platform IDs/URLs. Must contain 'anisearch_id'.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Anime data dict enriched with "episodes" and "characters" keys,
            or None if the primary anime fetch fails.
        """
        anisearch_id_str = ids.get("anisearch_id")
        if not anisearch_id_str:
            return None

        anisearch_id = int(anisearch_id_str)
        output_path = os.path.join(temp_dir, "anisearch.jsonl") if temp_dir else None

        logger.info(f"Fetching AniSearch data for ID {anisearch_id}")

        anime_data = await self.fetch_anime(anisearch_id)
        if not anime_data:
            return None

        if output_path:
            append_jsonl(output_path, anime_data)

        try:
            episode_data = await self.fetch_episodes(anisearch_id)
            if episode_data:
                anime_data["episodes"] = episode_data
                logger.info(f"Integrated {len(episode_data)} episodes")
        except Exception:
            logger.warning(
                f"Failed to fetch episodes for ID {anisearch_id}", exc_info=True
            )

        try:
            characters = await self.fetch_characters(anisearch_id)
            if characters:
                anime_data["characters"] = characters
                logger.info(f"Integrated {len(characters)} characters")
        except Exception:
            logger.warning(
                f"Failed to fetch characters for ID {anisearch_id}", exc_info=True
            )

        return anime_data

    async def fetch_anime(self, anisearch_id: int) -> dict[str, Any] | None:
        """Fetch anime metadata for the given AniSearch numeric ID.

        Parameters:
            anisearch_id (int): AniSearch anime ID (e.g. 18878 for Dandadan).

        Returns:
            Canonical anime dict if available, None otherwise.
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
        """Fetch episode data for a given AniSearch anime ID.

        Parameters:
            anisearch_id (int): AniSearch numeric anime ID.

        Returns:
            List of episode dicts if found, None otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch episode data for ID {anisearch_id}")
            episode_data = await fetch_anisearch_episodes(
                anime_id=str(anisearch_id),
                output_path=None,
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

    async def fetch_character_refs(self, anisearch_id: int) -> list[dict[str, str]]:
        """Fetch character refs (URL + role) from the anime characters list page.

        Parameters:
            anisearch_id (int): AniSearch numeric anime ID.

        Returns:
            List of {"url": str, "role": str} dicts. Empty list on failure.
        """
        logger.info(f"Fetching AniSearch character refs for ID {anisearch_id}")
        return await fetch_anisearch_character_refs(str(anisearch_id))

    async def fetch_characters(
        self, anisearch_id: int
    ) -> list[dict[str, Any]] | None:
        """Fetch full character detail pages for a given AniSearch anime ID.

        Internally fetches character refs (list page) then batch-fetches detail pages.

        Parameters:
            anisearch_id (int): AniSearch numeric ID of the anime.

        Returns:
            List of canonical character dicts, or None if no characters found.
        """
        try:
            logger.info(f"Fetching AniSearch character data for ID {anisearch_id}")

            refs = await self.fetch_character_refs(anisearch_id)
            if not refs:
                logger.debug(f"No character refs found for ID {anisearch_id}")
                return None

            characters = await fetch_anisearch_characters(refs)
            non_null = [c for c in characters if c is not None]
            if not non_null:
                return None

            logger.info(
                f"Successfully fetched {len(non_null)} characters for ID {anisearch_id}"
            )
            return non_null
        except Exception:
            logger.exception(f"Error fetching character data for ID {anisearch_id}")
            return None
