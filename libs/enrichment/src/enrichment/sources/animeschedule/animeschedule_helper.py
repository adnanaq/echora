#!/usr/bin/env python3
"""
AnimSchedule Data Helper
Fetches anime data from AnimSchedule API (async version).

AnimSchedule has no persistent per-anime ID, so lookups are title-search-based:
the helper searches by title and cross-validates the result against existing
source URLs from offline data.

Usage:
    # Programmatic usage via class
    from enrichment.sources.animeschedule.animeschedule_helper import AnimescheduleHelper
    helper = AnimescheduleHelper()
    data = await helper.fetch_all(ids, offline_data)

    # CLI usage
    python -m enrichment.api_helpers.animeschedule_helper "One Piece"
    python -m enrichment.api_helpers.animeschedule_helper "One Piece" --output temp/as.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

import aiohttp
from common.utils.jsonl_utils import append_jsonl
from http_cache.instance import http_cache_manager as _cache_manager

from enrichment.sources.animeschedule.animeschedule_mapper import (
    anime_from_animeschedule,
)
from enrichment.sources.animeschedule.animeschedule_models import AnimScheduleAnime
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError

from enrichment.sources.base.base_helper import BaseEnrichmentHelper, normalize_enrichment_payload

logger = logging.getLogger(__name__)

# AnimSchedule website keys that carry cross-source URLs (partial, no scheme)
_CROSS_SOURCE_KEYS = ("mal", "aniList", "kitsu", "animePlanet", "anidb")


def _match_by_sources(candidates: list[dict], sources: list[str]) -> dict | None:
    """Return the first candidate whose websites dict contains any of the given sources.

    AnimSchedule stores partial URLs (no scheme), e.g. ``"myanimelist.net/anime/21"``.
    We normalize our sources the same way by stripping the scheme before comparing.

    Args:
        candidates: Raw anime dicts from the AnimSchedule search response.
        sources: Full canonical source URLs to match against.

    Returns:
        The first matching candidate dict, or None if no match is found.
    """
    normalized = {
        s.removeprefix("https://").removeprefix("http://") for s in sources if s
    }

    for candidate in candidates:
        websites = candidate.get("websites", {})
        for key in _CROSS_SOURCE_KEYS:
            partial = websites.get(key)
            if not isinstance(partial, str) or not partial:
                continue
            # Support both ID-only ("myanimelist.net/anime/21") and full slug
            # ("myanimelist.net/anime/21/One_Piece") on either side — match if
            # either string is a prefix of the other.
            if any(
                partial.startswith(src) or src.startswith(partial) for src in normalized
            ):
                return candidate

    return None


class AnimescheduleHelper(BaseEnrichmentHelper):
    """Fetch AnimSchedule data for an anime by title search.

    AnimSchedule has no persistent per-anime ID, so this helper uses a
    different lookup strategy than other enrichment helpers: it searches by
    ``offline_data["title"]`` and validates the result by checking whether any
    source URL from ``offline_data["sources"]`` appears in the AnimSchedule
    response's ``websites`` dict. This is intentional — ``ids`` is unused.
    """

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch and map AnimSchedule data for an anime.

        Lookup strategy: searches by ``offline_data["title"]``, then cross-validates
        each search result against ``offline_data["sources"]`` by checking whether any
        known source URL appears in the result's ``websites`` dict. Returns the first
        match. Falls back to the first result when no sources are provided.

        Note: ``ids`` is intentionally unused — AnimSchedule has no platform ID system.

        Args:
            ids: Unused. AnimSchedule is title-search-based with no platform ID.
            offline_data: The original offline anime metadata. Must contain ``"title"``.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Canonical anime dict, or None if no match is found or title is missing.
        """
        search_term = offline_data.get("title", "")
        if not search_term:
            logger.warning(
                "AnimescheduleHelper.fetch_all: offline_data missing 'title' — cannot search AnimSchedule"
            )
            return None

        output_path = (
            os.path.join(temp_dir, "animeschedule.jsonl") if temp_dir else None
        )
        sources: list[str] = offline_data.get("sources", [])

        result = await self._search(
            search_term, sources=sources or None, output_path=output_path
        )
        return normalize_enrichment_payload(result)

    async def _search(
        self,
        search_term: str,
        sources: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Search AnimSchedule by title and return a canonical mapped dict.

        When ``sources`` is provided, all search results are checked against the
        cross-source URLs in each result's ``websites`` dict (mal, aniList, kitsu,
        animePlanet, anidb). The first result whose website links match any of the
        provided sources is returned. Falls back to the first result when no sources
        are given.

        Args:
            search_term: Anime title to search for.
            sources: Canonical source URLs to validate the result against.
            output_path: If provided, write the canonical mapped dict as JSONL to this path.

        Returns:
            Canonical anime dict if a match is found, None otherwise.
        """
        logger.info(f"Fetching AnimSchedule data for: {search_term}")

        search_url = f"https://animeschedule.net/api/v3/anime?q={search_term}"
        logger.debug(f"AnimSchedule search URL: {search_url}")

        try:
            async with _cache_manager.get_aiohttp_session("animeschedule") as session:
                async with session.get(search_url) as response:
                    response.raise_for_status()
                    search_results = await response.json()
        except aiohttp.ClientError as e:
            raise ServiceNetworkError(service="animeschedule", cause=e) from e
        except json.JSONDecodeError as e:
            raise ServiceParseError(service="animeschedule", cause=e) from e

        candidates: list[dict] = (search_results or {}).get("anime", [])
        if not candidates:
            logger.warning(f"AnimSchedule returned no results for: {search_term}")
            return None

        if sources:
            raw_data = _match_by_sources(candidates, sources)
            if raw_data is None:
                logger.warning(
                    f"No AnimSchedule result matched sources for: {search_term}"
                )
                return None
        else:
            raw_data = candidates[0]

        anime = AnimScheduleAnime.model_validate(raw_data)
        result = anime_from_animeschedule(anime)

        if output_path:
            append_jsonl(output_path, result)

        logger.info(f"AnimSchedule data fetched: {search_term}")
        return result


async def main() -> int:
    """CLI entry point for fetching AnimSchedule data.

    Returns:
        0 on success, 1 if no data found or an error occurred.
    """
    parser = argparse.ArgumentParser(
        description="Fetch anime data from AnimSchedule API."
    )
    parser.add_argument("search_term", type=str, help="Anime title to search for")
    parser.add_argument(
        "--output",
        type=str,
        default="animeschedule.jsonl",
        help="Output file path (default: animeschedule.jsonl in current directory)",
    )
    args = parser.parse_args()

    helper = AnimescheduleHelper()
    try:
        result = await helper._search(args.search_term, output_path=args.output)
    except Exception:
        logger.exception("Error fetching AnimSchedule data")
        return 1
    else:
        return 0 if result else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
