#!/usr/bin/env python3
"""
AnimSchedule Data Helper
Fetches anime data from AnimSchedule API (async version).

AnimSchedule has no ID of its own that other sites link to, but it filters its
anime list by MyAnimeList, AniList and AniDB IDs, so the helper looks the anime
up by the IDs in its source URLs. Only when none of them finds it does it search
by title, keeping the result only if it links back to one of those sources.

Usage:
    # Programmatic usage via class
    from enrichment.sources.animeschedule.animeschedule_helper import AnimescheduleHelper
    helper = AnimescheduleHelper(cache)
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
import re
import sys
from typing import Any

import aiohttp
from common.config import CacheConfig, RedisConfig
from common.utils.jsonl_utils import append_jsonl
from enrichment.sources.animeschedule.animeschedule_mapper import (
    anime_from_animeschedule,
)
from enrichment.sources.animeschedule.animeschedule_models import AnimScheduleAnime
from enrichment.sources.base.base_helper import (
    BaseEnrichmentHelper,
    normalize_enrichment_payload,
)
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError
from http_cache import HTTPCacheManager, configure_cache

logger = logging.getLogger(__name__)

_ANIME_LIST_URL = "https://animeschedule.net/api/v3/anime"

# AnimSchedule website keys that carry cross-source URLs (partial, no scheme)
_CROSS_SOURCE_KEYS = ("mal", "aniList", "kitsu", "animePlanet", "anidb")

# ID filters of AnimSchedule's anime list, by the source URL they take the ID
# from. Tried in order: AnimSchedule does not link every anime to all three.
_ID_FILTERS = (
    ("mal_url", "mal-ids"),
    ("anilist_url", "anilist-ids"),
    ("anidb_url", "anidb-ids"),
)
_ANIME_ID = re.compile(r"/anime/(\d+)")

# AnimSchedule answers 404 when no anime has the ID and 500 when a title search
# finds nothing.
_NO_MATCH_STATUSES = frozenset({404, 500})


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
    """Fetch AnimSchedule data for an anime by its provider IDs, or its title."""

    def __init__(self, cache: HTTPCacheManager) -> None:
        """Initialize AnimSchedule helper.

        Args:
            cache: Supplies the HTTP sessions, cached when caching is on.
        """
        self._cache = cache

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any] | None:
        """Fetch and map AnimSchedule data for an anime.

        Looks the anime up by the MyAnimeList, AniList and AniDB IDs in ``ids``;
        when none finds it, searches by ``offline_data["title"]``.

        Args:
            ids: Source URLs by provider (``mal_url``, ``anilist_url``,
                ``anidb_url``).
            offline_data: The offline anime entry; its ``title`` and
                ``sources`` are used for the title search.
            temp_dir: Optional directory for intermediate JSONL storage.
            fetch_characters: Accepted for interface compliance; AnimSchedule is anime-only.
            fetch_episodes: Accepted for interface compliance; AnimSchedule is anime-only.

        Returns:
            Canonical anime payload, or None when AnimSchedule has no such anime.
        """
        raw_data = await self._find_by_ids(ids) or await self._find_by_title(
            offline_data.get("title", ""), offline_data.get("sources", [])
        )
        if raw_data is None:
            return None
        output_path = (
            os.path.join(temp_dir, "animeschedule.jsonl") if temp_dir else None
        )
        return normalize_enrichment_payload(self._map(raw_data, output_path))

    async def _find_by_ids(self, ids: dict[str, str]) -> dict | None:
        """Return the anime AnimSchedule lists under the first ID that finds one.

        Args:
            ids: Source URLs by provider.

        Returns:
            The raw anime, or None when no ID finds one.
        """
        for url_key, id_filter in _ID_FILTERS:
            if match := _ANIME_ID.search(ids.get(url_key) or ""):
                candidates = await self._query({id_filter: match.group(1)})
                if candidates:
                    return candidates[0]
        return None

    async def _find_by_title(self, title: str, sources: list[str]) -> dict | None:
        """Search by title and return the result that links to one of ``sources``.

        Args:
            title: Anime title to search for.
            sources: Source URLs the result must link to; when empty, the
                first result is returned.

        Returns:
            The raw anime, or None when nothing matches.
        """
        if not title:
            return None
        candidates = await self._query({"q": title})
        raw_data = (
            _match_by_sources(candidates, sources)
            if sources
            else next(iter(candidates), None)
        )
        if raw_data is None:
            logger.warning(f"AnimSchedule has no anime matching: {title}")
        return raw_data

    async def _query(self, params: dict[str, str]) -> list[dict]:
        """Query AnimSchedule's anime list.

        Args:
            params: Filter or search parameters.

        Returns:
            The anime found; empty when there is no match.

        Raises:
            ServiceNetworkError: If the request fails.
            ServiceParseError: If the response is not JSON.
        """
        try:
            async with self._cache.get_aiohttp_session("animeschedule") as session:
                async with session.get(_ANIME_LIST_URL, params=params) as response:
                    if response.status in _NO_MATCH_STATUSES:
                        return []
                    response.raise_for_status()
                    found = await response.json()
        except aiohttp.ClientError as e:
            raise ServiceNetworkError(service="animeschedule", cause=e) from e
        except json.JSONDecodeError as e:
            raise ServiceParseError(service="animeschedule", cause=e) from e
        return (found or {}).get("anime", [])

    def _map(self, raw_data: dict, output_path: str | None) -> dict[str, Any]:
        """Validate and map a raw anime, appending it to ``output_path`` when given.

        Args:
            raw_data: Raw anime from AnimSchedule.
            output_path: JSONL file to append the mapped anime to.

        Returns:
            Canonical anime fields.
        """
        result = anime_from_animeschedule(AnimScheduleAnime.model_validate(raw_data))
        if output_path:
            append_jsonl(output_path, result)
        logger.info(f"AnimSchedule data fetched: {result.get('title')}")
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

    helper = AnimescheduleHelper(configure_cache(CacheConfig(), RedisConfig()))
    try:
        raw_data = await helper._find_by_title(args.search_term, [])
        if raw_data is None:
            return 1
        helper._map(raw_data, args.output)
    except Exception:
        logger.exception("Error fetching AnimSchedule data")
        return 1
    else:
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
