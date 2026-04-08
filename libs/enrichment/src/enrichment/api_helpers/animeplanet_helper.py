#!/usr/bin/env python3
"""
Anime-Planet Helper for AI Enrichment Integration

Helper function to fetch Anime-Planet data using the scraper for AI enrichment pipeline.
"""

import asyncio
import json
import logging
import os
import sys
from types import TracebackType
from typing import Any

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.utils.jsonl_utils import append_jsonl

from ..crawlers.anime_planet.anime_planet_anime_crawler import fetch_animeplanet_anime
from ..crawlers.anime_planet_character_crawler import fetch_animeplanet_characters
from ..mappers.animeplanet_mapper import anime_from_animeplanet

logger = logging.getLogger(__name__)


class AnimePlanetEnrichmentHelper:
    """Helper for Anime-Planet data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize Anime-Planet enrichment helper."""

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

        except Exception:
            logger.exception(f"Error fetching character data for slug '{slug}'")
            return None

    async def fetch_anime_data(
        self, url: str, include_characters: bool = True
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
            anime = await fetch_animeplanet_anime(url)
            if not anime:
                logger.warning(f"Crawler returned no data for '{url}'")
                return None
            anime_data = anime_from_animeplanet(anime)

            # Fetch character data if requested
            if include_characters:
                try:
                    slug = anime.slug
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
                    logger.warning(f"Failed to fetch characters for '{url}': {e}")

            logger.info(f"Successfully fetched anime data for '{url}'")
            return anime_data

        except Exception:
            logger.exception(f"Error fetching anime data for '{url}'")
            return None

    async def fetch_all(
        self,
        url: str,
        anime_output_path: str | None = None,
        characters_output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch Anime-Planet anime and character data for the given URL.

        Anime data is written to ``anime_output_path`` immediately after the anime
        fetch completes. Character data is written to ``characters_output_path`` after
        the character fetch completes. This mirrors the MAL helper pattern so each
        data type is persisted as soon as it is available.

        Parameters:
            url: Full Anime-Planet anime URL (e.g. "https://www.anime-planet.com/anime/dandadan").
            anime_output_path: If provided, anime data is written to this JSONL file
                immediately after the anime fetch, before characters are fetched.
            characters_output_path: If provided, character data is written to this
                JSONL file after the character fetch completes.

        Returns:
            Dict with keys ``anime`` and ``characters``, or None when the anime fetch fails.
        """
        try:
            anime = await fetch_animeplanet_anime(url)
            if not anime:
                logger.warning(f"Crawler returned no data for '{url}'")
                return None
            anime_data = anime_from_animeplanet(anime)

            if anime_output_path:
                append_jsonl(anime_output_path, anime_data)

            logger.info(f"Anime-Planet anime fetched: {anime_data.get('title', url)}")

            characters: list[dict[str, Any]] = []
            try:
                characters_data = await self.fetch_character_data(anime.slug)
                if characters_data:
                    characters = characters_data.get("characters", [])
                    if characters_output_path:
                        append_jsonl(characters_output_path, characters_data)
                    logger.info(
                        f"Anime-Planet characters fetched: {len(characters)} characters"
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch characters for '{url}': {e}")

            return {"anime": anime_data, "characters": characters}

        except Exception:
            logger.exception(f"Error in fetch_all for URL '{url}'")
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
    """CLI: fetch Anime-Planet data for a given URL and write to a JSON file.

    Usage:
        python animeplanet_helper.py <url> <output_file>

    Returns:
        0 on success, 1 on failure.
    """
    if len(sys.argv) != 3:
        print(
            "Usage: python animeplanet_helper.py <url> <output_file>", file=sys.stderr
        )
        return 1

    try:
        url = sys.argv[1]
        output_file = sys.argv[2]

        async with AnimePlanetEnrichmentHelper() as helper:
            data = await helper.fetch_anime_data(url)

            if data:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"Data for {url} saved to {output_file}")
                return 0
            else:
                print(f"Could not fetch data for {url}", file=sys.stderr)
                return 1
    except (OSError, ValueError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
