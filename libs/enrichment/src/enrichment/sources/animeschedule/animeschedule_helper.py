#!/usr/bin/env python3
"""
AnimeSchedule Data Helper
Fetches anime data from AnimeSchedule API (async version).

AnimeSchedule has no ID of its own that other sites link to, but it filters its
anime list by MyAnimeList, AniList and AniDB IDs, so the helper looks the anime
up by the IDs in its source URLs. Only when none of them finds it does it search
by title, keeping the result only if it links back to one of those sources.

The app token is sent with every request: without it, AnimeSchedule leaves the
provider links out of its answers, so no result can be matched to its anime.

Usage:
    # Programmatic usage via class
    from enrichment.sources.animeschedule.animeschedule_helper import AnimescheduleHelper
    helper = AnimescheduleHelper(cache, config.animeschedule_token)
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
from common.config import CacheConfig, EnrichmentConfig, RedisConfig
from common.utils.jsonl_utils import append_jsonl
from enrichment.sources.animeschedule.animeschedule_mapper import (
    anime_from_animeschedule,
)
from enrichment.sources.animeschedule.animeschedule_models import AnimeScheduleAnime
from enrichment.sources.base.base_helper import (
    BaseEnrichmentHelper,
    normalize_enrichment_payload,
)
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError
from http_cache import HTTPCacheManager, configure_cache
from pydantic import SecretStr

logger = logging.getLogger(__name__)

_ANIME_LIST_URL = "https://animeschedule.net/api/v3/anime"

# AnimeSchedule website keys that carry cross-source URLs (partial, no scheme)
_CROSS_SOURCE_KEYS = ("mal", "aniList", "kitsu", "animePlanet", "anidb")

# ID filters of AnimeSchedule's anime list, by the source URL they take the ID
# from. Tried in order: AnimeSchedule does not link every anime to all three.
_ID_FILTERS = (
    ("mal_url", "mal-ids"),
    ("anilist_url", "anilist-ids"),
    ("anidb_url", "anidb-ids"),
)
_ANIME_ID = re.compile(r"/anime/(\d+)")

# AnimeSchedule answers 404 when no anime has the ID, and when a title search
# finds nothing ("Couldn't find anime with these parameters."). Any other
# status, a 500 included, is a real failure, so a server error is never taken
# for a missing anime.
_NOT_FOUND_STATUSES = frozenset({404})


def _page_address(url: str) -> str:
    """Return a URL without its scheme or trailing slash, as AnimeSchedule stores links.

    >>> _page_address("https://anilist.co/anime/154587/Sousou-no-Frieren/")
    'anilist.co/anime/154587/Sousou-no-Frieren'
    """
    return url.removeprefix("https://").removeprefix("http://").rstrip("/")


def _same_page(link: str, source: str) -> bool:
    """Report whether two page addresses name the same anime.

    Either side may carry a title after the ID, so one may extend the other,
    but only at a ``/``: anime 2 is not anime 21.

    >>> _same_page("myanimelist.net/anime/21/One_Piece", "myanimelist.net/anime/21")
    True
    >>> _same_page("myanimelist.net/anime/2", "myanimelist.net/anime/21")
    False
    """
    return (
        link == source or link.startswith(f"{source}/") or source.startswith(f"{link}/")
    )


def _match_by_sources(candidates: list[dict], sources: list[str]) -> dict | None:
    """Return the first candidate whose websites dict contains any of the given sources.

    AnimeSchedule stores links without the scheme, e.g. ``"myanimelist.net/anime/21"``,
    so both sides are compared through ``_page_address``.

    Args:
        candidates: Raw anime dicts from the AnimeSchedule search response.
        sources: Full canonical source URLs to match against.

    Returns:
        The first matching candidate dict, or None if no match is found.
    """
    addresses = {_page_address(source) for source in sources if source}
    for candidate in candidates:
        websites = candidate.get("websites", {})
        for key in _CROSS_SOURCE_KEYS:
            link = websites.get(key)
            if isinstance(link, str) and link:
                link_address = _page_address(link)
                if any(_same_page(link_address, address) for address in addresses):
                    return candidate
    return None


class AnimescheduleHelper(BaseEnrichmentHelper):
    """Fetch AnimeSchedule data for an anime by its provider IDs, or its title."""

    def __init__(
        self, cache: HTTPCacheManager, app_token: SecretStr | None = None
    ) -> None:
        """Initialize AnimeSchedule helper.

        Args:
            cache: Supplies the HTTP sessions, cached when caching is on.
            app_token: The AnimeSchedule app token, sent as a bearer token with
                every request; None sends none, and the answers then lack the
                provider links.
        """
        self._cache = cache
        self._request_headers = (
            {"Authorization": f"Bearer {app_token.get_secret_value()}"}
            if app_token
            else {}
        )

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any] | None:
        """Fetch and map AnimeSchedule data for an anime.

        Looks the anime up by the MyAnimeList, AniList and AniDB IDs in ``ids``;
        when none finds it, searches by ``offline_data["title"]``.

        Args:
            ids: Source URLs by provider (``mal_url``, ``anilist_url``,
                ``anidb_url``).
            offline_data: The offline anime entry; its ``title`` and
                ``sources`` are used for the title search.
            temp_dir: Optional directory for intermediate JSONL storage.
            fetch_characters: Accepted for interface compliance; AnimeSchedule is anime-only.
            fetch_episodes: Accepted for interface compliance; AnimeSchedule is anime-only.

        Returns:
            Canonical anime payload, or None when AnimeSchedule has no such anime.
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

    async def fetch_by_title(
        self, title: str, output_path: str | None = None
    ) -> dict[str, Any] | None:
        """Search by title and map the first result.

        Args:
            title: Anime title to search for.
            output_path: JSONL file to append the mapped anime to.

        Returns:
            Canonical anime fields, or None when the search finds nothing.
        """
        raw_data = await self._find_by_title(title, [])
        return None if raw_data is None else self._map(raw_data, output_path)

    async def _find_by_ids(self, ids: dict[str, str]) -> dict | None:
        """Return the anime AnimeSchedule lists under the first ID that finds one.

        A result counts only when it links back to the source URL its ID came
        from.

        Args:
            ids: Source URLs by provider.

        Returns:
            The raw anime, or None when no ID finds one.
        """
        for url_key, id_filter in _ID_FILTERS:
            source_url = ids.get(url_key) or ""
            if match := _ANIME_ID.search(source_url):
                candidates = await self._query({id_filter: match.group(1)})
                if found := _match_by_sources(candidates, [source_url]):
                    return found
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
            logger.warning(f"AnimeSchedule has no anime matching: {title}")
        return raw_data

    async def _query(self, params: dict[str, str]) -> list[dict]:
        """Query AnimeSchedule's anime list.

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
                async with session.get(
                    _ANIME_LIST_URL, params=params, headers=self._request_headers
                ) as response:
                    if response.status in _NOT_FOUND_STATUSES:
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
            raw_data: Raw anime from AnimeSchedule.
            output_path: JSONL file to append the mapped anime to.

        Returns:
            Canonical anime fields.
        """
        result = anime_from_animeschedule(AnimeScheduleAnime.model_validate(raw_data))
        if output_path:
            append_jsonl(output_path, result)
        logger.info(f"AnimeSchedule data fetched: {result.get('title')}")
        return result


async def main() -> int:
    """CLI entry point for fetching AnimeSchedule data.

    Returns:
        0 on success, 1 if no data found or an error occurred.
    """
    parser = argparse.ArgumentParser(
        description="Fetch anime data from AnimeSchedule API."
    )
    parser.add_argument("search_term", type=str, help="Anime title to search for")
    parser.add_argument(
        "--output",
        type=str,
        default="animeschedule.jsonl",
        help="Output file path (default: animeschedule.jsonl in current directory)",
    )
    args = parser.parse_args()

    helper = AnimescheduleHelper(
        configure_cache(CacheConfig(), RedisConfig()),
        EnrichmentConfig().animeschedule_token,
    )
    try:
        result = await helper.fetch_by_title(args.search_term, args.output)
    except Exception:
        logger.exception("Error fetching AnimeSchedule data")
        return 1
    return 0 if result else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
