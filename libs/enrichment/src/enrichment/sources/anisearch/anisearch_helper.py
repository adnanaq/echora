"""AniSearch enrichment helper using crawlers (anime, episodes, characters)."""

import logging
import os
from typing import Any

from common.utils.jsonl_utils import append_jsonl

from enrichment.sources.anisearch.anisearch_anime_crawler import fetch_anisearch_anime
from enrichment.sources.anisearch.anisearch_character_crawler import fetch_anisearch_characters
from enrichment.sources.anisearch.anisearch_character_refs_crawler import (
    fetch_anisearch_character_refs,
)
from enrichment.crawlers.anisearch_episode_crawler import fetch_anisearch_episodes
from enrichment.sources.base.base_helper import BaseEnrichmentHelper, normalize_enrichment_payload

logger = logging.getLogger(__name__)


class AniSearchHelper(BaseEnrichmentHelper):
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
        url = ids.get("anisearch_url")
        if not url:
            return None

        output_path = os.path.join(temp_dir, "anisearch.jsonl") if temp_dir else None

        logger.info(f"Fetching AniSearch data for {url}")

        anime_data = await self.fetch_anime(url)
        if not anime_data:
            return None

        if output_path:
            append_jsonl(output_path, anime_data)

        # Use the canonical URL (with slug) resolved by the crawler on redirect.
        canonical_url = (anime_data.get("sources") or [url])[0]

        episode_data = []
        try:
            episode_data = await self.fetch_episodes(canonical_url)
            if episode_data:
                logger.info(f"Integrated {len(episode_data)} episodes")
        except Exception:
            logger.warning(f"Failed to fetch episodes for {canonical_url}", exc_info=True)

        characters = []
        try:
            characters = await self.fetch_characters(canonical_url)
            if characters:
                logger.info(f"Integrated {len(characters)} characters")
        except Exception:
            logger.warning(f"Failed to fetch characters for {canonical_url}", exc_info=True)

        return normalize_enrichment_payload(
            {
                "anime": anime_data,
                "episodes": episode_data or [],
                "characters": characters or [],
                "extras": {},
            }
        )

    async def fetch_anime(self, url: str) -> dict[str, Any] | None:
        """Fetch anime metadata from the given AniSearch URL.

        Parameters:
            url: Full AniSearch anime URL (e.g. https://www.anisearch.com/anime/18878,dan-da-dan).

        Returns:
            Canonical anime dict if available, None otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch anime data for {url}")
            anime_data = await fetch_anisearch_anime(url, output_path=None)
            if not anime_data:
                logger.warning(f"Anime crawler returned no data for {url}")
                return None
            logger.info(
                f"Successfully fetched anime data for {url}: "
                f"{anime_data.get('japanese_title', 'Unknown')}"
            )
            return anime_data
        except Exception:
            logger.exception(f"Error fetching anime data for {url}")
            return None

    async def fetch_episodes(self, url: str) -> list[dict[str, Any]] | None:
        """Fetch episode data for the given AniSearch anime URL.

        Parameters:
            url: Full AniSearch anime URL.

        Returns:
            List of episode dicts if found, None otherwise.
        """
        try:
            logger.info(f"Fetching AniSearch episode data for {url}")
            episode_data = await fetch_anisearch_episodes(anime_id=url, output_path=None)
            if not episode_data:
                logger.debug(f"No episode data found for {url}")
                return None
            logger.info(f"Successfully fetched {len(episode_data)} episodes for {url}")
            return episode_data
        except Exception:
            logger.exception(f"Error fetching episode data for {url}")
            return None

    async def fetch_character_refs(self, url: str) -> list[dict[str, str]]:
        """Fetch character refs (URL + role) from the anime characters list page.

        Parameters:
            url: Full AniSearch anime URL.

        Returns:
            List of {"url": str, "role": str} dicts. Empty list on failure.
        """
        logger.info(f"Fetching AniSearch character refs for {url}")
        return await fetch_anisearch_character_refs(url)

    async def fetch_characters(self, url: str) -> list[dict[str, Any]] | None:
        """Fetch full character detail pages for the given AniSearch anime URL.

        Internally fetches character refs (list page) then batch-fetches detail pages.

        Parameters:
            url: Full AniSearch anime URL.

        Returns:
            List of canonical character dicts, or None if no characters found.
        """
        try:
            logger.info(f"Fetching AniSearch character data for {url}")

            refs = await self.fetch_character_refs(url)
            if not refs:
                logger.debug(f"No character refs found for {url}")
                return None

            characters = await fetch_anisearch_characters(refs)
            non_null = [c for c in characters if c is not None]
            if not non_null:
                return None

            logger.info(f"Successfully fetched {len(non_null)} characters for {url}")
            return non_null
        except Exception:
            logger.exception(f"Error fetching character data for {url}")
            return None
