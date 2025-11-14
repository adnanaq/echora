#!/usr/bin/env python3
"""
Anime-Planet Helper for AI Enrichment Integration

Helper function to fetch Anime-Planet data using the scraper for AI enrichment pipeline.
"""

import asyncio
import json
import logging
import os
import re
import sys
from types import TracebackType
from typing import Any, Dict, Optional, Type

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ..crawlers.anime_planet_anime_crawler import fetch_animeplanet_anime
from ..crawlers.anime_planet_character_crawler import fetch_animeplanet_characters

logger = logging.getLogger(__name__)


class AnimePlanetEnrichmentHelper:
    """Helper for Anime-Planet data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize Anime-Planet enrichment helper."""

    async def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract Anime-Planet slug from URL."""
        try:
            # Pattern: https://www.anime-planet.com/anime/SLUG
            match = re.search(r"anime-planet\.com/anime/([^/?]+)", url)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            logger.error(f"Error extracting slug from URL {url}: {e}")
            return None

    async def find_animeplanet_url(
        self, offline_anime_data: Dict[str, Any]
    ) -> Optional[str]:
        """Find Anime-Planet URL from offline anime data sources."""
        try:
            sources = offline_anime_data.get("sources", [])
            for source in sources:
                if isinstance(source, str) and "anime-planet.com" in source:
                    return source
            return None
        except Exception as e:
            logger.error(f"Error finding Anime-Planet URL: {e}")
            return None

    async def fetch_character_data(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch character data by slug using the new character crawler.

        Args:
            slug: Anime-Planet anime slug (e.g., 'dandadan')

        Returns:
            Dict containing character data with keys:
            - characters: List of character dicts with name, role, image_url, url
            - total_count: Total number of characters
        """
        try:
            character_data = await fetch_animeplanet_characters(
                slug=slug,
                return_data=True,
                output_path=None,  # No file output - return data only
            )

            if not character_data:
                logger.warning(f"Character crawler returned no data for slug '{slug}'")
                return None

            logger.info(
                f"Successfully fetched {character_data.get('total_count', 0)} characters "
                f"for '{slug}' using crawler"
            )
            return character_data

        except Exception as e:
            logger.error(f"Error fetching character data for slug '{slug}': {e}")
            return None

    async def fetch_anime_data(
        self, slug: str, include_characters: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch anime data by slug using the new crawler.

        Args:
            slug: Anime-Planet anime slug (e.g., 'dandadan')
            include_characters: Whether to fetch and include character data (default: True)

        Crawler returns comprehensive anime data including:
        - Basic info (title, description, slug, url)
        - Metadata (rank, studios, genres, episodes, year, season, status)
        - Related anime (same franchise)
        - Images (poster, image)
        - Characters (if include_characters=True)
        """
        try:
            anime_data = await fetch_animeplanet_anime(
                slug=slug,
                return_data=True,
                output_path=None,  # No file output - return data only
            )

            if not anime_data:
                logger.warning(f"Crawler returned no data for slug '{slug}'")
                return None

            # Fetch character data if requested
            if include_characters:
                try:
                    characters_data = await self.fetch_character_data(slug)
                    if characters_data:
                        anime_data["characters"] = characters_data.get("characters", [])
                        anime_data["character_count"] = characters_data.get(
                            "total_count", 0
                        )
                        logger.info(
                            f"Integrated {anime_data['character_count']} characters "
                            f"into anime data for '{slug}'"
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch characters for '{slug}': {e}")
                    # Continue without characters data - non-critical

            logger.info(f"Successfully fetched anime data for '{slug}' using crawler")
            return anime_data

        except Exception as e:
            logger.error(f"Error fetching anime data for slug '{slug}': {e}")
            return None

    async def fetch_all_data(
        self, offline_anime_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch all Anime-Planet data for an anime.

        Args:
            offline_anime_data: The offline anime data containing sources

        Returns:
            Dict containing Anime-Planet data or None if not found

        Note:
            Requires Anime-Planet URL in offline_anime_data["sources"].
            Title-based search is not supported by the crawler.
        """
        try:
            # Try to find direct URL in sources
            animeplanet_url = await self.find_animeplanet_url(offline_anime_data)

            if animeplanet_url:
                # Extract slug from URL
                slug = await self.extract_slug_from_url(animeplanet_url)
                if slug:
                    logger.info(f"Found Anime-Planet slug: {slug}")
                    return await self.fetch_anime_data(slug)
                else:
                    logger.error(
                        f"Failed to extract slug from Anime-Planet URL: {animeplanet_url}"
                    )
                    return None

            # No Anime-Planet URL found in sources
            title = offline_anime_data.get("title", "Unknown")
            logger.debug(
                f"No Anime-Planet URL in sources for '{title}', skipping enrichment"
            )
            return None

        except Exception as e:
            title = offline_anime_data.get("title", "Unknown")
            logger.error(
                f"Error in fetch_all_data for anime '{title}': {e}", exc_info=True
            )
            return None

    async def close(self) -> None:
        """No persistent session to close (creates per-request sessions)."""
        pass

    async def __aenter__(self) -> "AnimePlanetEnrichmentHelper":
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Exit async context."""
        await self.close()
        return False


async def main() -> int:
    """CLI entry point for AnimePlanet helper."""
    if len(sys.argv) != 3:
        print(
            "Usage: python animeplanet_helper.py <slug> <output_file>", file=sys.stderr
        )
        return 1

    try:
        slug = sys.argv[1]
        output_file = sys.argv[2]

        helper = AnimePlanetEnrichmentHelper()
        data = await helper.fetch_anime_data(slug)

        if data:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data for {slug} saved to {output_file}")
            return 0
        else:
            print(f"Could not fetch data for {slug}", file=sys.stderr)
            return 1
    except (OSError, ValueError, KeyError) as e:
        # OSError: File I/O failures
        # ValueError: Invalid JSON encoding
        # KeyError: Missing expected data fields
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
