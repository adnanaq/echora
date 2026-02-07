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
from typing import Any

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ..crawlers.anime_planet_anime_crawler import fetch_animeplanet_anime
from ..crawlers.anime_planet_character_crawler import fetch_animeplanet_characters

logger = logging.getLogger(__name__)


class AnimePlanetEnrichmentHelper:
    """Helper for Anime-Planet data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize Anime-Planet enrichment helper."""

    async def extract_slug_from_url(self, url: str) -> str | None:
        """Extract Anime-Planet slug from URL."""
        try:
            # Pattern: https://www.anime-planet.com/anime/SLUG
            match = re.search(r"anime-planet\.com/anime/([^/?]+)", url)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            logger.exception("Error extracting slug from URL {url}: ")
            return None

    async def find_animeplanet_url(
        self, offline_anime_data: dict[str, Any]
    ) -> str | None:
        """
        Locate the first Anime-Planet URL in the provided offline anime record.

        Searches the record's "sources" list for a string that contains "anime-planet.com" and returns the first match.

        Parameters:
            offline_anime_data (Dict[str, Any]): Offline anime record; expected to include a "sources" sequence of source entries.

        Returns:
            Optional[str]: The first source string containing "anime-planet.com", or `None` if no such source is found.
        """
        try:
            sources = offline_anime_data.get("sources", [])
            for source in sources:
                if isinstance(source, str) and "anime-planet.com" in source:
                    return source
            return None
        except Exception as e:
            logger.exception("Error finding Anime-Planet URL: ")
            return None

    async def fetch_character_data(self, slug: str) -> dict[str, Any] | None:
        """
        Retrieve character data for an Anime-Planet anime slug.

        Parameters:
            slug (str): Anime-Planet anime slug (for example, "dandadan").

        Returns:
            dict or None: A dictionary containing:
                - "characters": list of character dictionaries (each with keys such as "name", "role", "image_url", "url")
                - "total_count": integer total number of characters
            Returns `None` if no data is available or an error occurs.
        """
        try:
            character_data = await fetch_animeplanet_characters(
                slug=slug,
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
            logger.exception("Error fetching character data for slug '{slug}': ")
            return None

    async def fetch_anime_data(
        self, slug: str, include_characters: bool = True
    ) -> dict[str, Any] | None:
        """
        Fetch comprehensive Anime-Planet data for the given anime slug.

        Parameters:
            slug (str): Anime-Planet anime slug (e.g., "dandadan").
            include_characters (bool): If True, attempt to fetch and include character data.

        Returns:
            dict or None: A dictionary containing anime data (e.g., title, description, slug, url,
            metadata such as rank/studios/genres/episodes/year/season/status, related anime, images).
            If `include_characters` is True and available, the result will also include `characters`
            (list) and `character_count` (int). Returns `None` if no data could be retrieved.
        """
        try:
            anime_data = await fetch_animeplanet_anime(
                slug=slug,
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
            logger.exception("Error fetching anime data for slug '{slug}': ")
            return None

    async def fetch_all_data(
        self, offline_anime_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Locate an Anime-Planet URL in offline data, extract its slug, and fetch the corresponding Anime-Planet data.

        Parameters:
            offline_anime_data (Dict[str, Any]): Offline anime record expected to include a "sources" list and optionally a "title" for logging.

        Returns:
            Dict[str, Any] or None: The assembled Anime-Planet data (possibly including characters) if a usable URL and slug are found, `None` otherwise.

        Notes:
            This operation requires a direct Anime-Planet URL to be present in `offline_anime_data["sources"]`; title-based lookups are not performed.
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
        """
        Perform any necessary cleanup; this is a no-op because the helper holds no persistent resources.
        """
        pass

    async def __aenter__(self) -> "AnimePlanetEnrichmentHelper":
        """
        Provide async context manager entry that yields the helper instance.

        Returns:
            AnimePlanetEnrichmentHelper: The AnimePlanetEnrichmentHelper instance to be used within the async context.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """
        Exit the asynchronous context manager and perform any necessary cleanup.

        Returns:
            bool: `False` to indicate that any exception raised in the context should not be suppressed.
        """
        await self.close()
        return False


async def main() -> int:
    """
    Run the CLI that fetches Anime-Planet data for a given slug and writes it to the specified output file.

    Returns:
        exit_code (int): 0 on success (data was fetched and written), 1 on failure (invalid usage, fetch/write error, or missing data).
    """
    if len(sys.argv) != 3:
        print(
            "Usage: python animeplanet_helper.py <slug> <output_file>", file=sys.stderr
        )
        return 1

    try:
        slug = sys.argv[1]
        output_file = sys.argv[2]

        async with AnimePlanetEnrichmentHelper() as helper:
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
