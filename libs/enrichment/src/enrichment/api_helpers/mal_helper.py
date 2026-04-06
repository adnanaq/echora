#!/usr/bin/env python3

"""MAL enrichment helper (direct scraping via Crawl4AI).

Replaces the old Jikan v4 API integration with direct MAL scraping using
the crawlers in enrichment.crawlers.mal_crawler. The public interface is preserved so
that api_fetcher.py and stage scripts need no changes.

CLI:
  python -m enrichment.api_helpers.mal_helper anime <anime_url> <output_file>
  python -m enrichment.api_helpers.mal_helper episodes <anime_url> <episode_count> <output_file>
  python -m enrichment.api_helpers.mal_helper characters <anime_url> <output_file>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from common.utils.jsonl_utils import append_jsonl as _append_jsonl

from enrichment.crawlers.mal_crawler.mal_anime_crawler import fetch_mal_anime
from enrichment.crawlers.mal_crawler.mal_base import normalize_mal_anime_url
from enrichment.crawlers.mal_crawler.mal_character_crawler import (
    fetch_mal_character,
    fetch_mal_characters,
)
from enrichment.crawlers.mal_crawler.mal_character_refs_crawler import (
    fetch_mal_character_refs,
)
from enrichment.crawlers.mal_crawler.mal_episode_count_crawler import (
    fetch_mal_episode_count,
)
from enrichment.crawlers.mal_crawler.mal_episode_crawler import fetch_mal_episodes
from enrichment.mappers.mal_mapper import (
    anime_from_mal,
    character_from_mal,
    episode_from_mal,
)

logger = logging.getLogger(__name__)


class MalEnrichmentHelper:
    """Fetch MAL data via direct scraping for a single anime.

    Keeps the same public interface as the old Jikan-based helper so that
    api_fetcher.py and stage scripts require no changes.

    Args:
        mal_source: Full MAL anime URL (e.g. "https://myanimelist.net/anime/21").
        session: Ignored — kept for interface compatibility with the old helper.
            The crawlers manage their own browser sessions internally.
    """

    def __init__(
        self,
        mal_source: str,
        *,
        session: Any | None = None,  # noqa: ARG002  — compatibility only
    ) -> None:
        _, has_slug = normalize_mal_anime_url(
            mal_source
        )  # raises ValueError on invalid input
        self._mal_source = mal_source
        self._anime_url: str = mal_source if has_slug else ""

    async def close(self) -> None:
        """No-op — crawlers manage their own sessions."""

    async def __aenter__(self) -> MalEnrichmentHelper:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False

    async def fetch_anime(self) -> dict[str, Any] | None:
        """Fetch anime detail from /anime/{id}.

        If the anime page reports an unknown episode count (e.g. ongoing series),
        the count is resolved from the episode list page and patched into the result.

        Returns:
            MalScrapedAnime as a JSON-serializable dict, or None on failure.
        """
        anime = await fetch_mal_anime(self._mal_source)
        if anime is None:
            return None
        if not self._anime_url:
            self._anime_url = anime.source
        anime_dict = anime_from_mal(anime)
        if not anime_dict.get("episode_count") and self._anime_url:
            anime_dict["episode_count"] = await fetch_mal_episode_count(self._anime_url)
        return anime_dict

    async def fetch_character_urls(self) -> list[str]:
        """Fetch all character URLs from /anime/{id}/characters.

        A single page fetch returns all characters (1475 for One Piece).
        Requires fetch_anime() to have been called first so that _anime_url
        is set to the canonical slug URL (e.g. /anime/57334/Dandadan).

        Returns:
            List of character URLs. Empty on failure.
        """
        if not self._anime_url:
            logger.error(
                "fetch_character_urls called before fetch_anime — anime URL unknown"
            )
            return []
        return await fetch_mal_character_refs(f"{self._anime_url}/characters")

    async def fetch_episodes(
        self, episode_count: int, *, output_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch episode details in a single Docker batch job for episodes 1..episode_count.

        Uncached URLs are submitted as one crawl4ai batch job; already-cached
        URLs are returned directly from Redis without touching Docker.
        Concurrency is controlled by the Docker server's MAX_CONCURRENT_TASKS.

        Args:
            episode_count: Number of episodes to fetch. Use fetch_anime() first to
                get the correct count — it resolves "Unknown" counts automatically.
            output_path: If provided, each episode is appended to this JSONL file
                as it is parsed (streaming write).

        Returns:
            List of episode dicts (failed episodes are skipped).
        """
        if not self._anime_url:
            logger.error(
                "fetch_episodes requires a slug URL — pass https://myanimelist.net/anime/{id}/{slug}"
            )
            return []
        if episode_count == 0:
            return []
        urls = [f"{self._anime_url}/episode/{i}" for i in range(1, episode_count + 1)]

        _path = output_path

        def _on_episode(ep: Any) -> None:
            if _path:
                _append_jsonl(_path, episode_from_mal(ep))

        episodes_or_none = await fetch_mal_episodes(
            urls, on_result=_on_episode if output_path else None
        )
        return [episode_from_mal(ep) for ep in episodes_or_none if ep is not None]

    async def fetch_character(
        self,
        url: str,
        *,
        output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch detailed character data from /character/{id}.

        Args:
            url: Full MAL character URL (e.g. https://myanimelist.net/character/40/Luffy).
            output_path: If provided, the character is appended to this JSONL file.

        Returns:
            MalScrapedCharacter as a dict, or None on failure.
        """
        char = await fetch_mal_character(url)
        result = character_from_mal(char) if char else None
        if result and output_path:
            _append_jsonl(output_path, result)
        return result

    async def fetch_characters(
        self,
        urls: list[str],
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch detailed character data in a single batch Docker job.

        Args:
            urls: List of full MAL character URLs from fetch_character_urls().
            output_path: If provided, each character is appended to this JSONL file
                as it arrives (streaming write).

        Returns:
            List of detailed character dicts (failed ones are skipped).
        """

        _path = output_path

        def _on_character(c: Any) -> None:
            if _path:
                _append_jsonl(_path, character_from_mal(c))

        chars = await fetch_mal_characters(
            urls, on_result=_on_character if output_path else None
        )
        return [character_from_mal(c) for c in chars if c is not None]

    async def fetch_all(
        self,
        *,
        anime_output_path: str | None = None,
        episodes_output_path: str | None = None,
        characters_output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch all MAL data for this anime.

        Args:
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
        logger.info(f"Fetching MAL data for: {self._mal_source}")
        anime_info = await self.fetch_anime()
        if not anime_info:
            return None

        if anime_output_path:
            _append_jsonl(anime_output_path, anime_info)

        logger.info(f"MAL anime fetched: {anime_info.get('title', self._mal_source)}")

        episode_count = int(anime_info.get("episode_count") or 0)

        async def _fetch_episodes() -> list[dict[str, Any]]:
            if episode_count == 0:
                return []
            try:
                return await self.fetch_episodes(
                    episode_count, output_path=episodes_output_path
                )
            except Exception as e:
                logger.warning(
                    f"Episode fetch failed, continuing without episodes: {e}"
                )
                return []

        async def _fetch_characters() -> list[dict[str, Any]]:
            urls = await self.fetch_character_urls()
            if not urls:
                return []
            try:
                return await self.fetch_characters(
                    urls, output_path=characters_output_path
                )
            except Exception as e:
                logger.warning(
                    f"Character detail fetch failed, continuing without characters: {e}"
                )
                return []

        episodes_data = await _fetch_episodes()
        logger.info(f"MAL episodes fetched: {len(episodes_data)} episodes")
        characters_data = await _fetch_characters()
        logger.info(f"MAL characters fetched: {len(characters_data)} characters")
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Fetch data from MAL via direct scraping"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anime = sub.add_parser("anime", help="Fetch anime detail page")
    p_anime.add_argument(
        "anime_url", help="Full MAL anime URL (e.g. https://myanimelist.net/anime/21)"
    )
    p_anime.add_argument("output_file", help="Output JSON file")

    p_eps = sub.add_parser("episodes", help="Fetch episode detail pages")
    p_eps.add_argument(
        "anime_url", help="Full MAL anime URL (e.g. https://myanimelist.net/anime/21)"
    )
    p_eps.add_argument("episode_count", type=int, help="Number of episodes to fetch")
    p_eps.add_argument("output_file", help="Output JSON file")

    p_chars = sub.add_parser("characters", help="Fetch character detail pages")
    p_chars.add_argument(
        "anime_url", help="Full MAL anime URL (e.g. https://myanimelist.net/anime/21)"
    )
    p_chars.add_argument("output_file", help="Output JSON file")

    args = parser.parse_args()

    async with MalEnrichmentHelper(args.anime_url) as helper:
        if args.cmd == "anime":
            data = await helper.fetch_anime()
            _write_json_sync(args.output_file, data or {})
            return 0

        if args.cmd == "episodes":
            episodes = await helper.fetch_episodes(args.episode_count)
            _write_json_sync(args.output_file, episodes)
            return 0

        if args.cmd == "characters":
            basic = await helper.fetch_character_urls()
            detailed = await helper.fetch_characters(basic)
            _write_json_sync(args.output_file, detailed)
            return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
