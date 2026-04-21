#!/usr/bin/env python3
"""
Anime-Planet Helper for AI Enrichment Integration

Helper function to fetch Anime-Planet data using the scraper for AI enrichment pipeline.

CLI:
  python -m enrichment.api_helpers.anime_planet_helper anime <url> --output <anime.jsonl>
  python -m enrichment.api_helpers.anime_planet_helper characters <url1> [url2 ...] --output <chars.jsonl>
  python -m enrichment.api_helpers.anime_planet_helper all <url> [--anime-output <anime.jsonl>] [--chars-output <chars.jsonl>]
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

from common.utils.jsonl_utils import append_jsonl

from ..crawlers.anime_planet.anime_planet_anime_crawler import fetch_animeplanet_anime
from ..crawlers.anime_planet.anime_planet_character_crawler import (
    fetch_animeplanet_characters,
)
from ..crawlers.anime_planet.anime_planet_character_refs_crawler import (
    fetch_animeplanet_character_refs,
)
from .base_helper import BaseEnrichmentHelper

logger = logging.getLogger(__name__)

_AP_BASE_URL = "https://www.anime-planet.com"


def _normalize_ap_url(url: str) -> str:
    """Normalize an AP URL to the canonical www form.

    The offline database stores AP URLs without the www subdomain
    (e.g. https://anime-planet.com/anime/dandadan).  The live site
    redirects to www, so we normalize upfront to avoid an extra hop.
    """
    return url.replace("https://anime-planet.com/", "https://www.anime-planet.com/")


class AnimePlanetEnrichmentHelper(BaseEnrichmentHelper):
    """Helper for Anime-Planet data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize Anime-Planet enrichment helper."""

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch Anime-Planet anime and character data for the given URL.

        Anime data is written to ``anime_output_path`` immediately after the anime
        fetch completes. Character data is written to ``characters_output_path`` after
        the character fetch completes. This mirrors the MAL helper pattern so each
        data type is persisted as soon as it is available.

        Args:
            ids: Dictionary of validated platform IDs/URLs. Must contain 'anime_planet_url'.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Dict with keys ``anime`` and ``characters``, or None when the anime fetch fails.
        """
        url = ids.get("anime_planet_url")
        if not url:
            return None

        anime_output_path = (
            os.path.join(temp_dir, "anime_planet_anime.jsonl") if temp_dir else None
        )
        characters_output_path = (
            os.path.join(temp_dir, "anime_planet_characters.jsonl")
            if temp_dir
            else None
        )

        canonical_url = _normalize_ap_url(url)
        try:
            anime = await fetch_animeplanet_anime(canonical_url)
            if not anime:
                logger.warning(f"Crawler returned no data for '{canonical_url}'")
                return None
            anime_data = anime

            if anime_output_path:
                append_jsonl(anime_output_path, anime_data)

            logger.info(
                f"Anime-Planet anime fetched: {anime_data.get('title', canonical_url)}"
            )

            characters: list[dict[str, Any]] = []
            try:
                anime_url = (anime_data.get("sources") or [""])[0] or canonical_url
                characters = await self.fetch_characters(
                    anime_url, output_path=characters_output_path
                )
                logger.info(
                    f"Anime-Planet characters fetched: {len(characters)} characters"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch characters for '{canonical_url}': {e}")

        except Exception:
            logger.exception(f"Error in fetch_all for URL '{url}'")
            return None
        else:
            return {"anime": anime_data, "characters": characters}

    async def fetch_characters(
        self,
        anime_url: str,
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all characters for an Anime-Planet anime URL.

        Two-step flow: refs crawl (URL per character) → batch detail crawl.
        Each character is written to ``output_path`` immediately via on_result callback
        (write-immediately pattern, mirrors MAL helper).

        Args:
            anime_url: Full Anime-Planet anime URL (e.g. "https://www.anime-planet.com/anime/dandadan").
                Non-www URLs are normalized to www before the crawl.
            output_path: If provided, each mapped character is appended to this JSONL
                file as it completes.

        Returns:
            List of canonical Character dicts. Empty on failure.
        """
        canonical_url = _normalize_ap_url(anime_url)
        characters_url = f"{canonical_url}/characters"
        try:
            refs = await fetch_animeplanet_character_refs(characters_url)
            if not refs:
                logger.warning(f"No character refs for '{canonical_url}'")
                return []

            urls = [f"{_AP_BASE_URL}{ref['url']}" for ref in refs]
            results: list[dict[str, Any]] = []

            _path = output_path

            def _on_character(char: dict[str, Any]) -> None:
                results.append(char)
                if _path:
                    append_jsonl(_path, char)

            await fetch_animeplanet_characters(urls, on_result=_on_character)

            logger.info(f"Fetched {len(results)} characters for '{canonical_url}'")
        except Exception:
            logger.exception(f"Error fetching character data for '{canonical_url}'")
            return []
        else:
            return results

    async def fetch_anime(self, url: str) -> dict[str, Any] | None:
        """Fetch Anime-Planet anime data for the given URL.

        Args:
            url: Anime-Planet anime URL (e.g., "https://anime-planet.com/anime/dandadan").
                Non-www URLs are normalized to www before the crawl.

        Returns:
            Canonical anime dict, or None if no data could be retrieved.
        """
        canonical_url = _normalize_ap_url(url)
        try:
            anime = await fetch_animeplanet_anime(canonical_url)
            if not anime:
                logger.warning(f"Crawler returned no data for '{canonical_url}'")
                return None
            logger.info(f"Successfully fetched anime data for '{canonical_url}'")
        except Exception:
            logger.exception(f"Error fetching anime data for '{canonical_url}'")
            return None
        else:
            return anime

async def main() -> int:
    """CLI entry point for fetching Anime-Planet data."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Fetch data from Anime-Planet via direct scraping"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anime = sub.add_parser("anime", help="Fetch anime detail page")
    p_anime.add_argument(
        "url",
        help="Full Anime-Planet anime URL (e.g. https://www.anime-planet.com/anime/dandadan)",
    )
    p_anime.add_argument("--output", required=True, help="Output JSONL file")

    p_chars = sub.add_parser("characters", help="Fetch character detail pages")
    p_chars.add_argument(
        "urls",
        nargs="+",
        help="One or more full character URLs (e.g. https://www.anime-planet.com/characters/luffy)",
    )
    p_chars.add_argument("--output", required=True, help="Output JSONL file")

    p_all = sub.add_parser("all", help="Fetch anime and all its characters")
    p_all.add_argument(
        "url",
        help="Full Anime-Planet anime URL (e.g. https://www.anime-planet.com/anime/dandadan)",
    )
    p_all.add_argument(
        "--anime-output", default=None, help="Output JSONL file for anime data"
    )
    p_all.add_argument(
        "--chars-output", default=None, help="Output JSONL file for character data"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        return 1

    helper = AnimePlanetEnrichmentHelper()

    if args.cmd == "anime":
        data = await helper.fetch_anime(args.url)
        if data is None:
            logger.error(f"No data for '{args.url}'")
            return 1
        append_jsonl(args.output, data)
        return 0

    if args.cmd == "characters":
        out_path = args.output

        def _on_char(char: dict[str, Any]) -> None:
            append_jsonl(out_path, char)

        await fetch_animeplanet_characters(args.urls, on_result=_on_char)
        return 0

    if args.cmd == "all":
        ids = {"anime_planet_url": args.url}
        result = await helper.fetch_all(
            ids,
            {},
            None,  # temp_dir not used in CLI for all output paths
        )
        return 0 if result is not None else 1

    return 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
