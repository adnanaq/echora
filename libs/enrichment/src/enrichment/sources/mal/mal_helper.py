#!/usr/bin/env python3

"""MAL enrichment helper (direct scraping via Crawl4AI)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from enrichment.sources.base.base_helper import (
    BaseEnrichmentHelper,
    normalize_enrichment_payload,
)
from enrichment.sources.mal.mal_anime_crawler import fetch_mal_anime
from enrichment.sources.mal.mal_base import normalize_mal_anime_url
from enrichment.sources.mal.mal_character_crawler import (
    fetch_mal_character,
    fetch_mal_characters,
)
from enrichment.sources.mal.mal_character_refs_crawler import (
    fetch_mal_character_refs,
)
from enrichment.sources.mal.mal_episode_count_crawler import (
    fetch_mal_episode_count,
)
from enrichment.sources.mal.mal_episode_crawler import fetch_mal_episodes

logger = logging.getLogger(__name__)


class MalHelper(BaseEnrichmentHelper):
    """Fetch MAL data via direct scraping for a single anime."""

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch all MAL data for this anime.

        Args:
            ids: Dictionary of validated platform IDs/URLs. Must contain 'mal_url'.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Dict with keys ``anime``, ``episodes``, ``characters``, or None when
            the anime fetch fails.
        """
        url = ids.get("mal_url")
        if not url:
            return None

        try:
            _, has_slug = normalize_mal_anime_url(url)
        except ValueError:
            logger.warning(f"Invalid MAL URL: {url}")
            return None

        if not has_slug:
            logger.warning(f"MAL URL must include slug: {url}")
            return None

        anime_output_path = (
            os.path.join(temp_dir, "mal_anime.jsonl") if temp_dir else None
        )
        episodes_output_path = (
            os.path.join(temp_dir, "mal_episodes.jsonl") if temp_dir else None
        )
        characters_output_path = (
            os.path.join(temp_dir, "mal_characters.jsonl") if temp_dir else None
        )

        logger.info(f"Fetching MAL data for: {url}")
        anime_info = await self._fetch_anime(url, output_path=anime_output_path)
        if not anime_info:
            return None

        logger.info(f"MAL anime fetched: {anime_info.get('title', url)}")

        episode_count = int(anime_info.get("episode_count") or 0)

        episodes_data: list[dict[str, Any]] = []
        if episode_count > 0:
            try:
                episodes_data = await self._fetch_episodes(
                    url, episode_count, output_path=episodes_output_path
                )
            except Exception as e:
                logger.warning(
                    f"Episode fetch failed, continuing without episodes: {e}"
                )

        characters_data: list[dict[str, Any]] = []
        try:
            characters_data = await self._fetch_characters(
                url, output_path=characters_output_path
            )
        except Exception as e:
            logger.warning(
                f"Character detail fetch failed, continuing without characters: {e}"
            )

        logger.info(f"MAL episodes fetched: {len(episodes_data)} episodes")
        logger.info(f"MAL characters fetched: {len(characters_data)} characters")

        return normalize_enrichment_payload(
            {
                "anime": anime_info,
                "episodes": episodes_data,
                "characters": characters_data,
            }
        )

    async def _fetch_anime(
        self, url: str, *, output_path: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch anime detail from MAL.

        Args:
            url: Full MAL anime slug URL.
            output_path: If provided, write result as a JSONL line to this file.

        Returns:
            Canonical anime dict, or None on failure.
        """
        anime = await fetch_mal_anime(url, output_path=output_path)
        if anime is None:
            return None
        if not anime.get("episode_count"):
            anime["episode_count"] = await fetch_mal_episode_count(url)
        return anime

    async def _fetch_character_urls(self, anime_url: str) -> list[str]:
        """Fetch all character URLs for an anime.

        Args:
            anime_url: Full MAL anime slug URL.

        Returns:
            List of character URLs. Empty on failure.
        """
        return await fetch_mal_character_refs(f"{anime_url}/characters")

    async def _fetch_episodes(
        self,
        anime_url: str,
        episode_count: int,
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch episode details for episodes 1..episode_count.

        Args:
            anime_url: Full MAL anime URL (slug or numeric).
            episode_count: Number of episodes to fetch.
            output_path: If provided, each episode is appended to this JSONL file.

        Returns:
            List of episode dicts (failed episodes are skipped).
        """
        if episode_count == 0:
            return []
        urls = [f"{anime_url}/episode/{i}" for i in range(1, episode_count + 1)]
        episodes_or_none = await fetch_mal_episodes(urls, output_path=output_path)
        return [ep for ep in episodes_or_none if ep is not None]

    async def _fetch_character(
        self,
        url: str,
        *,
        output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch detailed character data from /character/{id}.

        Args:
            url: Full MAL character URL.
            output_path: If provided, the character is appended to this JSONL file.

        Returns:
            MalCharacter as a dict, or None on failure.
        """
        return await fetch_mal_character(url, output_path=output_path)

    async def _fetch_characters(
        self,
        anime_url: str,
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all characters for an anime.

        Args:
            anime_url: Full MAL anime slug URL.
            output_path: If provided, each character is appended to this JSONL file.

        Returns:
            List of detailed character dicts (failed ones are skipped).
        """
        urls = await self._fetch_character_urls(anime_url)
        if not urls:
            return []
        chars = await fetch_mal_characters(urls, output_path=output_path)
        return [c for c in chars if c is not None]


def _write_json_sync(path: str, data: Any) -> None:
    """Write a JSON document to disk synchronously (CLI use only)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main() -> int:
    """CLI entrypoint for fetching MAL data."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Fetch data from MAL via direct scraping"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anime = sub.add_parser("anime", help="Fetch anime detail page")
    p_anime.add_argument(
        "anime_url",
        help="Full MAL anime URL (e.g. https://myanimelist.net/anime/21/One_Piece)",
    )
    p_anime.add_argument("output_file", help="Output JSON file")

    p_eps = sub.add_parser("episodes", help="Fetch episode detail pages")
    p_eps.add_argument(
        "anime_url",
        help="Full MAL anime URL (e.g. https://myanimelist.net/anime/21/One_Piece)",
    )
    p_eps.add_argument("episode_count", type=int, help="Number of episodes to fetch")
    p_eps.add_argument("output_file", help="Output JSON file")

    p_chars = sub.add_parser("characters", help="Fetch character detail pages")
    p_chars.add_argument(
        "anime_url",
        help="Full MAL anime URL (e.g. https://myanimelist.net/anime/21/One_Piece)",
    )
    p_chars.add_argument("output_file", help="Output JSON file")

    args = parser.parse_args()

    helper = MalHelper()
    if args.cmd == "anime":
        result = await helper.fetch_all({"mal_url": args.anime_url}, {}, None)
        _write_json_sync(args.output_file, result or {})
        return 0

    if args.cmd == "episodes":
        episodes = await helper._fetch_episodes(args.anime_url, args.episode_count)
        _write_json_sync(args.output_file, episodes)
        return 0

    if args.cmd == "characters":
        chars = await helper._fetch_characters(args.anime_url)
        _write_json_sync(args.output_file, chars)
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
