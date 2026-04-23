#!/usr/bin/env python3

"""MAL enrichment helper (direct scraping via Crawl4AI).

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

from enrichment.sources.base.base_helper import BaseEnrichmentHelper, normalize_enrichment_payload

logger = logging.getLogger(__name__)


class MalHelper(BaseEnrichmentHelper):
    """Fetch MAL data via direct scraping for a single anime."""

    def __init__(self) -> None:
        """Initialize MAL helper."""
        self._mal_source = ""
        self._anime_url: str = ""

    def _setup_url(self, url: str) -> None:
        """Set source URL and derive the slug URL used for episode/character fetches.

        Args:
            url: Full MAL anime URL (e.g. ``https://myanimelist.net/anime/21``).

        Raises:
            ValueError: If ``url`` is not a valid MAL anime URL.
        """
        _, has_slug = normalize_mal_anime_url(url)
        self._mal_source = url
        self._anime_url = url if has_slug else ""

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
            self._setup_url(url)
        except ValueError:
            logger.warning(f"Invalid MAL URL: {url}")
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

        episodes_data, characters_data = await asyncio.gather(
            _fetch_episodes(), _fetch_characters()
        )

        logger.info(f"MAL episodes fetched: {len(episodes_data)} episodes")
        logger.info(f"MAL characters fetched: {len(characters_data)} characters")

        return normalize_enrichment_payload({
            "anime": anime_info,
            "episodes": episodes_data,
            "characters": characters_data,
        })

    async def fetch_anime(self) -> dict[str, Any] | None:
        """Fetch anime detail from /anime/{id}.

        If the anime page reports an unknown episode count (e.g. ongoing series),
        the count is resolved from the episode list page and patched into the result.

        Returns:
            MalAnime as a JSON-serializable dict, or None on failure.
        """
        anime = await fetch_mal_anime(self._mal_source)
        if anime is None:
            return None
        if not self._anime_url:
            self._anime_url = (anime.get("sources") or [""])[0]
        if not anime.get("episode_count") and self._anime_url:
            anime["episode_count"] = await fetch_mal_episode_count(self._anime_url)
        return anime

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

        def _on_episode(ep: dict[str, Any]) -> None:
            if _path:
                _append_jsonl(_path, ep)

        episodes_or_none = await fetch_mal_episodes(
            urls, on_result=_on_episode if output_path else None
        )
        return [ep for ep in episodes_or_none if ep is not None]

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
            MalCharacter as a dict, or None on failure.
        """
        result = await fetch_mal_character(url)
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

        def _on_character(c: dict[str, Any]) -> None:
            if _path:
                _append_jsonl(_path, c)

        chars = await fetch_mal_characters(
            urls, on_result=_on_character if output_path else None
        )
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

    helper = MalHelper()
    ids = {"mal_url": args.anime_url}
    if args.cmd == "anime":
        result = await helper.fetch_all(ids, {}, None)
        _write_json_sync(args.output_file, result or {})
        return 0

    if args.cmd == "episodes":
        helper._setup_url(args.anime_url)
        episodes = await helper.fetch_episodes(args.episode_count)
        _write_json_sync(args.output_file, episodes)
        return 0

    if args.cmd == "characters":
        helper._setup_url(args.anime_url)
        basic = await helper.fetch_character_urls()
        detailed = await helper.fetch_characters(basic)
        _write_json_sync(args.output_file, detailed)
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
