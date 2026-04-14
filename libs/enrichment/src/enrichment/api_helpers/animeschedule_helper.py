#!/usr/bin/env python3
"""
AnimSchedule Data Helper
Fetches anime data from AnimSchedule API (async version).

Usage:
    # Programmatic usage (no file output)
    from enrichment.api_helpers.animeschedule_helper import fetch_all
    data = await fetch_all("One Piece", sources=["https://myanimelist.net/anime/21"])

    # Programmatic usage (with file output)
    data = await fetch_all("One Piece", output_path="temp/animeschedule.jsonl")

    # CLI usage (default output to CWD)
    python -m enrichment.api_helpers.animeschedule_helper "One Piece"
    # Output: animeschedule.jsonl

    # CLI usage (custom output path)
    python -m enrichment.api_helpers.animeschedule_helper "One Piece" --output temp/as.jsonl
    # Output: temp/as.jsonl
"""

import argparse
import asyncio
import json
import logging
import sys

import aiohttp
from common.utils.jsonl_utils import append_jsonl
from http_cache.instance import http_cache_manager as _cache_manager

from enrichment.api_helpers.animeschedule.animeschedule_models import AnimScheduleAnime
from enrichment.exceptions import ServiceNetworkError, ServiceParseError
from enrichment.api_helpers.animeschedule.animeschedule_mapper import anime_from_animeschedule

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


class AnimescheduleEnrichmentHelper:
    """Class wrapper around fetch_all for registry-based initialization in api_fetcher."""

    async def fetch_all(
        self,
        search_term: str,
        sources: list[str] | None = None,
        output_path: str | None = None,
    ) -> dict | None:
        """Delegate to the module-level :func:`fetch_all`.

        Args:
            search_term: Anime title to search for.
            sources: Canonical source URLs to validate the result against.
            output_path: If provided, write the result as JSONL to this path.

        Returns:
            Canonical anime dict, or None if no match is found.
        """
        return await fetch_all(search_term, sources=sources, output_path=output_path)

    async def close(self) -> None:
        """No-op — sessions are created per-call inside fetch_all."""
        pass  # session is created and closed per-call inside fetch_all


async def fetch_all(
    search_term: str,
    sources: list[str] | None = None,
    output_path: str | None = None,
) -> dict | None:
    """Fetch and map AnimSchedule data for an anime by title.

    When ``sources`` is provided, all search results are checked against the
    cross-source URLs in each result's ``websites`` dict (mal, aniList, kitsu,
    animePlanet, anidb). The first result whose website links match any of the
    provided sources is returned — even if the search returned only one result.
    Falls back to the first result only when no sources are given.

    Args:
        search_term: Anime title to search for.
        sources: Canonical source URLs to validate the result against.
        output_path: If provided, write the canonical mapped dict as JSONL to this path.

    Returns:
        Canonical anime dict if a match is found, None otherwise.
    """
    logger.info(f"Fetching AnimSchedule data for: {search_term}")

    session = _cache_manager.get_aiohttp_session("animeschedule")

    try:
        search_url = f"https://animeschedule.net/api/v3/anime?q={search_term}"
        logger.debug(f"AnimSchedule search URL: {search_url}")

        async with session.get(search_url) as response:
            response.raise_for_status()
            search_results = await response.json()

        candidates: list[dict] = (search_results or {}).get("anime", [])
        if not candidates:
            logger.warning(f"AnimSchedule returned no results for: {search_term}")
            return None

        if sources:
            raw_data = _match_by_sources(candidates, sources)
            if raw_data is None:
                logger.warning(f"No AnimSchedule result matched sources for: {search_term}")
                return None
        else:
            raw_data = candidates[0]

        anime = AnimScheduleAnime.model_validate(raw_data)
        result = anime_from_animeschedule(anime)

        if output_path:
            append_jsonl(output_path, result)

        logger.info(f"AnimSchedule data fetched: {search_term}")
        return result

    except aiohttp.ClientError as e:
        raise ServiceNetworkError(service="animeschedule", cause=e) from e
    except json.JSONDecodeError as e:
        raise ServiceParseError(service="animeschedule", cause=e) from e
    finally:
        await session.close()


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

    try:
        result = await fetch_all(args.search_term, output_path=args.output)
        return 0 if result else 1
    except Exception:
        logger.exception("Error fetching AnimSchedule data")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
