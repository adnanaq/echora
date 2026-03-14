#!/usr/bin/env python3

"""MAL enrichment helper (direct scraping via Crawl4AI).

Replaces the old Jikan v4 API integration with direct MAL scraping using
the crawlers in enrichment.crawlers.mal_crawler. The public interface is preserved so
that api_fetcher.py and stage scripts need no changes.

CLI:
  python -m enrichment.api_helpers.mal_helper anime <mal_id> <output_file>
  python -m enrichment.api_helpers.mal_helper episodes <mal_id> <episode_count> <output_file>
  python -m enrichment.api_helpers.mal_helper characters <mal_id> <output_file>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from enrichment.api_helpers.mal_rate_limiter import (
    MalRateLimiter,
    get_shared_mal_rate_limiter,
)
from enrichment.crawlers.mal_crawler.mal_anime_crawler import fetch_mal_anime
from enrichment.crawlers.mal_crawler.mal_character_crawler import (
    fetch_mal_character,
    fetch_mal_characters,
)
from enrichment.crawlers.mal_crawler.mal_character_refs_crawler import fetch_mal_character_refs
from enrichment.crawlers.mal_crawler.mal_episode_crawler import (
    fetch_mal_episode,
    fetch_mal_episodes,
)
from enrichment.mappers.mal_mapper import anime_from_mal, character_from_mal, episode_from_mal

logger = logging.getLogger(__name__)


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a single JSON record as a JSONL line. Logs and continues on failure."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    except Exception as e:
        logger.warning("Failed to append record to %s: %s", path, e)


class MalEnrichmentHelper:
    """Fetch MAL data via direct scraping for a single anime.

    Keeps the same public interface as the old Jikan-based helper so that
    api_fetcher.py and stage scripts require no changes.

    Args:
        anime_id: MyAnimeList anime ID (stringified or int).
        session: Ignored — kept for interface compatibility with the old helper.
            The crawlers manage their own browser sessions internally.
        limiter: Optional rate limiter override.
    """

    def __init__(
        self,
        anime_id: str,
        *,
        session: Any | None = None,  # noqa: ARG002  — compatibility only
        limiter: MalRateLimiter | None = None,
    ) -> None:
        self._anime_id = str(anime_id)
        self._mal_id = int(anime_id)
        self._anime_url: str = ""
        self._limiter = limiter or get_shared_mal_rate_limiter()

    async def close(self) -> None:
        """No-op — crawlers manage their own sessions."""

    async def __aenter__(self) -> MalEnrichmentHelper:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False

    async def fetch_anime(self) -> dict[str, Any] | None:
        """Fetch anime detail from /anime/{id}.

        Returns:
            MalScrapedAnime as a JSON-serializable dict, or None on failure.
        """
        anime = await fetch_mal_anime(self._mal_id)
        if anime is None:
            return None
        self._anime_url = anime.url
        return anime_from_mal(anime)

    async def fetch_characters_basic(self) -> list[dict[str, Any]]:
        """Fetch all character references from /anime/{id}/characters.

        A single page fetch returns all characters (1475 for One Piece).
        Requires fetch_anime() to have been called first so that _anime_url
        is set to the canonical slug URL (e.g. /anime/57334/Dandadan).

        Returns:
            List of CharacterRef dicts. Empty on failure.
        """
        if not self._anime_url:
            logger.error("fetch_characters_basic called before fetch_anime — anime URL unknown")
            return []
        refs = await fetch_mal_character_refs(self._mal_id, self._anime_url)
        return [ref.model_dump(mode="json") for ref in refs]

    async def fetch_episode_detail(self, episode_id: int) -> dict[str, Any] | None:
        """Fetch a single episode detail from /anime/{id}/episode/{num}.

        Args:
            episode_id: 1-based episode number.

        Returns:
            MalScrapedEpisode as a dict, or None on failure.
        """
        url = f"{self._anime_url}/episode/{episode_id}"
        ep = await fetch_mal_episode(url)
        if ep is None:
            return None
        result = ep.model_dump(mode="json")
        # Ensure episode_number is present for pipeline compatibility
        result["episode_number"] = episode_id
        return result

    async def fetch_episodes(
        self, episode_count: int, *, output_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch episode details in a single Docker batch job for episodes 1..episode_count.

        Uncached URLs are submitted as one crawl4ai batch job; already-cached
        URLs are returned directly from Redis without touching Docker.
        Concurrency is controlled by the Docker server's MAX_CONCURRENT_TASKS.

        Args:
            episode_count: Number of episodes to fetch.
            output_path: If provided, each episode is appended to this JSONL file
                as it is parsed (streaming write).

        Returns:
            List of episode dicts (failed episodes are skipped).
        """
        urls = [f"{self._anime_url}/episode/{i}" for i in range(1, episode_count + 1)]
        episodes_or_none = await fetch_mal_episodes(urls, output_path=output_path)
        return [
            episode_from_mal(ep)
            for ep in episodes_or_none
            if ep is not None
        ]

    async def fetch_character_detail(
        self, character_basic: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Fetch detailed character data from /character/{id}.

        Args:
            character_basic: CharacterRef dict from fetch_characters_basic().

        Returns:
            MalScrapedCharacter as a dict with injected role context, or None.
        """
        char_id: int | None = character_basic.get("char_id")
        if not isinstance(char_id, int):
            return None

        char = await fetch_mal_character(char_id)
        if char is None:
            return None

        # Name from character list ("Monkey D., Luffy") is in canonical "Last, First" order.
        # The detail page may use a different order, so prefer the list-page name.
        char.name = character_basic.get("name") or char.name

        return character_from_mal(char)

    async def fetch_characters_detailed(
        self,
        characters_basic: list[dict[str, Any]],
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch detailed character data in a single batch Docker job.

        Args:
            characters_basic: Output of fetch_characters_basic().
            output_path: If provided, each character is appended to this JSONL file
                as it arrives (streaming write).

        Returns:
            List of detailed character dicts (failed ones are skipped).
        """
        # Build char_id list and lookup for context injection
        char_ids: list[int] = []
        basic_by_id: dict[int, dict[str, Any]] = {}
        for item in characters_basic:
            char_id: int | None = item.get("char_id")
            if isinstance(char_id, int):
                char_ids.append(char_id)
                basic_by_id[char_id] = item

        chars = await fetch_mal_characters(char_ids)

        characters: list[dict[str, Any]] = []
        for char_id, char in zip(char_ids, chars):
            if char is None:
                continue
            # Name from character list is in canonical "Last, First" order — prefer it.
            basic = basic_by_id[char_id]
            char.name = basic.get("name") or char.name
            detail = character_from_mal(char)
            characters.append(detail)
            if output_path:
                _append_jsonl(output_path, detail)
        return characters

    async def fetch_all_data(
        self,
        *,
        fallback_episode_count: int = 0,
        anime_output_path: str | None = None,
        episodes_output_path: str | None = None,
        characters_output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch all MAL data for this anime.

        Args:
            fallback_episode_count: Episode count fallback when anime page lacks count.
            anime_output_path: If provided, anime data is written to this JSONL file
                immediately after the anime page is fetched.
            episodes_output_path: If provided, each episode is streamed to this JSONL
                file as it is parsed rather than held entirely in memory.
            characters_output_path: If provided, each character is streamed to this
                JSONL file as it arrives rather than held entirely in memory.

        Returns:
            Dict with keys ``anime``, ``episodes``, ``characters``, or None when
            the anime fetch fails.
        """
        anime_info = await self.fetch_anime()
        if not anime_info:
            return None

        if anime_output_path:
            _append_jsonl(anime_output_path, anime_info)

        episode_count = anime_info.get("episode_count") or fallback_episode_count
        episode_count = int(episode_count or 0)

        characters_basic = await self.fetch_characters_basic()
        episodes_data: list[dict[str, Any]] = []
        characters_data: list[dict[str, Any]] = characters_basic

        if episode_count > 0:
            try:
                episodes_data = await self.fetch_episodes(
                    episode_count, output_path=episodes_output_path
                )
            except Exception as e:
                logger.warning("Episode fetch failed, continuing without episodes: %s", e)

        if characters_basic:
            try:
                detailed_chars = await self.fetch_characters_detailed(
                    characters_basic, output_path=characters_output_path
                )
                characters_data = detailed_chars or characters_basic
            except Exception as e:
                logger.warning("Character detail fetch failed, using basic data: %s", e)

        return {
            "anime": anime_info,
            "episodes": episodes_data,
            "characters": characters_data,
        }


def _write_json_sync(path: str, data: Any) -> None:
    """Write a JSON document to disk synchronously (CLI use only)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main() -> int:
    """CLI entrypoint for fetching MAL data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch data from MAL via direct scraping")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anime = sub.add_parser("anime", help="Fetch anime detail page")
    p_anime.add_argument("anime_id", help="MyAnimeList anime ID")
    p_anime.add_argument("output_file", help="Output JSON file")

    p_eps = sub.add_parser("episodes", help="Fetch episode detail pages")
    p_eps.add_argument("anime_id", help="MyAnimeList anime ID")
    p_eps.add_argument("episode_count", type=int, help="Number of episodes to fetch")
    p_eps.add_argument("output_file", help="Output JSON file")

    p_chars = sub.add_parser("characters", help="Fetch character detail pages")
    p_chars.add_argument("anime_id", help="MyAnimeList anime ID")
    p_chars.add_argument("output_file", help="Output JSON file")

    args = parser.parse_args()

    async with MalEnrichmentHelper(args.anime_id) as helper:
        if args.cmd == "anime":
            data = await helper.fetch_anime()
            _write_json_sync(args.output_file, data or {})
            return 0

        if args.cmd == "episodes":
            episodes = await helper.fetch_episodes(args.episode_count)
            _write_json_sync(args.output_file, episodes)
            return 0

        if args.cmd == "characters":
            basic = await helper.fetch_characters_basic()
            detailed = await helper.fetch_characters_detailed(basic)
            _write_json_sync(args.output_file, detailed)
            return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
