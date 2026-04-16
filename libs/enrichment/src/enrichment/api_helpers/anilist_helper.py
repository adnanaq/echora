#!/usr/bin/env python3
"""
AniList Helper for AI Enrichment Integration

Production helper for fetching and enriching anime data from the AniList GraphQL API.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from types import TracebackType
from typing import Any

import aiohttp
from common.utils.jsonl_utils import append_jsonl
from common.utils.retry import retry_with_backoff
from http_cache.instance import http_cache_manager

from ..exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)
from .anilist.anilist_mapper import anime_from_anilist, character_from_anilist
from .anilist.anilist_anime_models import AniListAnime
from .anilist.anilist_character_models import AniListCharacterEdge
from .base_helper import BaseEnrichmentHelper

logger = logging.getLogger(__name__)

_ANILIST_BASE = "https://anilist.co/anime"


def _extract_anilist_id(url: str) -> int:
    """Extract numeric AniList ID from a URL like https://anilist.co/anime/21."""
    try:
        return int(url.rstrip("/").split("/")[-1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Cannot extract AniList ID from URL: {url!r}") from e


class AniListHelper(BaseEnrichmentHelper):
    """Helper for AniList data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize AniList helper."""
        self.base_url = "https://graphql.anilist.co"
        self.session: aiohttp.ClientSession | None = None
        self.rate_limit_remaining = 90
        self._session_event_loop: asyncio.AbstractEventLoop | None = None

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch canonical anime data and all characters for an AniList URL.

        Args:
            ids: Dictionary of validated platform IDs/URLs. Must contain 'anilist_url'.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Dict with keys ``anime`` and ``characters``, or ``None`` on failure.
        """
        url = ids.get("anilist_url")
        if not url:
            return None

        logger.info(f"Fetching AniList data for: {url}")
        anime = await self.fetch_anime_canonical(url, temp_dir)
        if anime:
            logger.info(f"AniList anime fetched: {anime.get('title', url)}")
        characters = await self.fetch_characters_canonical(url, temp_dir)
        logger.info(f"AniList characters fetched: {len(characters)} characters")

        if not anime and not characters:
            return None

        return {"anime": anime, "characters": characters}

    async def _ensure_session(self) -> None:
        """Lazily create (or recreate) the aiohttp session for the current event loop."""
        current_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        if self.session is None or self._session_event_loop != current_loop:
            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    logger.debug(
                        "Ignoring error while closing old session", exc_info=True
                    )
            self.session = http_cache_manager.get_aiohttp_session(
                "anilist",
                timeout=aiohttp.ClientTimeout(total=None),
            )
            logger.debug("AniList cached session created for current event loop")
            self._session_event_loop = current_loop

        if self.session is None:
            raise RuntimeError("Failed to initialize AniList session")

    async def _execute_request(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Single-attempt HTTP call to AniList GraphQL endpoint.

        Handles 429 with Retry-After (up to 3 waits before raising).
        Raises canonical ServiceError subtypes — no retry logic here.

        Raises:
            ServiceRateLimitedError: 429 exhausted.
            ServiceBlockedError: 403 — API disabled/blocked.
            AniListGraphQLError: HTTP 200/400 with GraphQL errors[] in body.
            ServiceNetworkError: Any other network / JSON decode failure.
        """
        await self._ensure_session()
        assert self.session is not None  # guaranteed by _ensure_session

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Hishel-Body-Key": "true",
        }
        payload = {"query": query, "variables": variables or {}}

        max_rate_limit_waits = 3
        for attempt in range(max_rate_limit_waits):
            try:
                async with self.session.post(
                    self.base_url, json=payload, headers=headers
                ) as response:
                    from_cache = getattr(response, "from_cache", False)

                    if "X-RateLimit-Remaining" in response.headers:
                        self.rate_limit_remaining = int(
                            response.headers["X-RateLimit-Remaining"]
                        )

                    if response.status == 429:
                        if attempt < max_rate_limit_waits - 1:
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(
                                f"AniList rate limit (attempt {attempt + 1}/{max_rate_limit_waits}). "
                                f"Waiting {retry_after}s..."
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        raise ServiceRateLimitedError(
                            service="anilist", attempts=max_rate_limit_waits
                        )

                    if response.status == 403:
                        raise ServiceBlockedError(
                            "API disabled/blocked", service="anilist"
                        )

                    response.raise_for_status()
                    data: Any = await response.json()

                    if "errors" in data:
                        logger.error(f"AniList GraphQL errors: {data['errors']}")
                        raise AniListGraphQLError(data["errors"])

                    result: dict[str, Any] = data.get("data", {})
                    result["_from_cache"] = from_cache

                    # Client-side throttle when approaching rate limit
                    if not from_cache and self.rate_limit_remaining < 5:
                        logger.warning(
                            f"Rate limit low ({self.rate_limit_remaining}), sleeping 60s."
                        )
                        await asyncio.sleep(60)
                        self.rate_limit_remaining = 90

                    return result

            except (ServiceRateLimitedError, ServiceBlockedError, AniListGraphQLError):
                raise
            except aiohttp.ClientResponseError as exc:
                # Non-429/403 4xx are not retryable
                if exc.status < 500:
                    raise ServiceNetworkError(service="anilist", cause=exc) from exc
                raise
            except (aiohttp.ClientError, json.JSONDecodeError, TimeoutError) as exc:
                raise ServiceNetworkError(service="anilist", cause=exc) from exc

        # Unreachable — loop always raises or returns
        raise AssertionError(
            "AniList rate-limit wait loop exited without raising or returning"
        )  # pragma: no cover

    async def _make_request(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a GraphQL request with retry on transient network errors.

        Raises:
            ServiceRateLimitedError: 429 exhausted after Retry-After waits.
            ServiceBlockedError: 403 API disabled.
            AniListGraphQLError: GraphQL errors in response body.
            ServiceNetworkError: Transient network failure after all retries.
        """
        return await retry_with_backoff(
            operation=lambda: self._execute_request(query, variables),
            max_retries=3,
            retry_delay=1.0,
            is_transient_error=lambda e: isinstance(
                e, (aiohttp.ClientError, json.JSONDecodeError, TimeoutError)
            )
            and not isinstance(e, aiohttp.ClientResponseError),
        )

    def _get_media_query_fields(self) -> str:
        """Return the GraphQL selection set for a Media (ANIME) node.

        Returns:
            GraphQL field selection string covering titles, images, stats, relations,
            studios, external links, tags, rankings, and airing info.
        """
        return """
        id
        idMal
        title {
          romaji
          english
          native
          userPreferred
        }
        description(asHtml: false)
        source
        format
        episodes
        duration
        status
        season
        seasonYear
        countryOfOrigin
        isAdult
        coverImage {
          extraLarge
          large
          medium
          color
        }
        bannerImage
        trailer {
          id
          site
          thumbnail
        }
        averageScore
        popularity
        favourites
        genres
        synonyms
        tags {
          name
          description
          category
          isAdult
        }
        relations {
          edges {
            node {
              id
              idMal
              title {
                romaji
                english
              }
              format
              status
              seasonYear
              averageScore
              coverImage { extraLarge }
              episodes
              chapters
              volumes
            }
            relationType
          }
        }
        studios {
          edges {
            node {
              id
              name
              isAnimationStudio
            }
            isMain
          }
        }
        externalLinks {
          id
          url
          site
          type
          language
        }
        nextAiringEpisode {
          episode
          airingAt
          timeUntilAiring
        }
        rankings {
          rank
          context
          format
          year
          season
          allTime
        }
        """

    def _build_query_by_mal_id(self) -> str:
        return f"query ($idMal: Int) {{ Media(idMal: $idMal, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    def _build_query_by_anilist_id(self) -> str:
        return f"query ($id: Int) {{ Media(id: $id, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    async def fetch_anime_by_mal_id(self, mal_id: int) -> dict[str, Any] | None:
        """Fetch raw AniList Media data by MyAnimeList ID.

        Args:
            mal_id: MyAnimeList integer anime ID.

        Returns:
            Raw ``Media`` dict from AniList, or None if not found.
        """
        query = self._build_query_by_mal_id()
        variables = {"idMal": mal_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def fetch_anime(self, anilist_id: int) -> dict[str, Any] | None:
        """Fetch raw AniList Media data by AniList ID.

        Args:
            anilist_id: AniList integer anime ID.

        Returns:
            Raw ``Media`` dict from AniList, or None if not found.
        """
        query = self._build_query_by_anilist_id()
        variables = {"id": anilist_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def _fetch_paginated_data(
        self, anilist_id: int, query_template: str, data_key: str
    ) -> list[dict[str, Any]]:
        """Fetch all paginated edges for a Media sub-resource.

        Args:
            anilist_id: AniList numeric ID of the Media node.
            query_template: GraphQL query accepting ``id`` and ``page`` variables,
                returning a paginated structure under ``Media``.
            data_key: Key under ``Media`` whose container provides ``edges`` and
                ``pageInfo`` (e.g. ``"characters"``, ``"staff"``).

        Returns:
            All edge objects across all pages. Cached pages skip the inter-page delay.
        """
        all_items = []
        page = 1
        has_next_page = True
        while has_next_page:
            variables = {"id": anilist_id, "page": page}
            response = await self._make_request(query_template, variables)
            if (
                not response
                or not response.get("Media")
                or not response["Media"].get(data_key)
            ):
                break
            data = response["Media"][data_key]
            all_items.extend(data.get("edges", []))
            has_next_page = data.get("pageInfo", {}).get("hasNextPage", False)
            page += 1

            # Only rate limit for network requests, not cache hits, and only between pages (not after the last)
            if not response.get("_from_cache", False) and has_next_page:
                await asyncio.sleep(0.5)

        return all_items

    async def fetch_characters(self, anilist_id: int) -> list[dict[str, Any]]:
        """Fetch all character edges for an anime from AniList.

        Args:
            anilist_id: AniList numeric ID of the anime.

        Returns:
            List of edge dicts from the ``characters`` connection; each edge
            contains ``node`` (character data), ``role``, and ``voiceActors``.
        """
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            characters(page: $page, perPage: 50, sort: ROLE) {
              pageInfo { hasNextPage }
              edges {
                node {
                  id
                  name {
                    full
                    native
                    alternative
                    alternativeSpoiler
                  }
                  image { large medium }
                  description
                  gender
                  dateOfBirth { year month day }
                  age
                  bloodType
                  favourites
                  siteUrl
                }
                role
                voiceActorRoles {
                  voiceActor {
                    id
                    name { full native }
                    languageV2
                    image { large }
                    siteUrl
                  }
                  roleNotes
                  dubGroup
                }
              }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "characters")

    async def fetch_anime_canonical(
        self, url: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch anime data, validate via AniListAnime, and map to canonical fields.

        Args:
            url: Full AniList anime URL (e.g. ``https://anilist.co/anime/21``).
            temp_dir: If provided, write canonical JSONL to
                ``{temp_dir}/anilist.jsonl``.

        Returns:
            Canonical dict from ``anime_from_anilist``, or None if not found.
        """
        anilist_id = _extract_anilist_id(url)
        raw = await self.fetch_anime(anilist_id)
        if not raw:
            logger.warning(f"No AniList data found for: {url}")
            return None

        anime = AniListAnime.model_validate(raw)
        canonical = anime_from_anilist(anime)

        if temp_dir:
            append_jsonl(os.path.join(temp_dir, "anilist.jsonl"), canonical)

        return canonical

    async def fetch_characters_canonical(
        self, url: str, temp_dir: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all character edges, validate, and map to canonical dicts.

        Args:
            url: Full AniList anime URL (e.g. ``https://anilist.co/anime/21``).
            temp_dir: If provided, append each character as JSONL to
                ``{temp_dir}/anilist_characters.jsonl``.

        Returns:
            List of canonical character dicts from ``character_from_anilist``.
        """
        anilist_id = _extract_anilist_id(url)
        raw_edges = await self.fetch_characters(anilist_id)
        if not raw_edges:
            return []

        out_path = (
            os.path.join(temp_dir, "anilist_characters.jsonl") if temp_dir else None
        )
        canonical = []
        for edge in raw_edges:
            try:
                char_edge = AniListCharacterEdge.model_validate(edge)
                char = character_from_anilist(char_edge)
                canonical.append(char)
                if out_path:
                    append_jsonl(out_path, char)
            except Exception:
                logger.warning("Skipping invalid character edge", exc_info=True)

        return canonical

    async def _fetch_all_data_by_mal_id(self, mal_id: int) -> dict[str, Any] | None:
        """Fetch raw anime and character data by MyAnimeList ID.

        Args:
            mal_id: MyAnimeList integer anime ID.

        Returns:
            Raw Media dict with ``characters`` key populated, or None if not found.
        """
        anime_data = await self.fetch_anime_by_mal_id(mal_id)
        if not anime_data:
            logger.warning(f"No AniList data found for MAL ID: {mal_id}")
            return None
        characters = await self.fetch_characters(anime_data["id"])
        if characters:
            anime_data["characters"] = {"edges": characters}
        return anime_data

    async def close(self) -> None:
        """Close the active aiohttp session if one exists."""
        if self.session:
            await self.session.close()
            # Reset session state to prevent accidental reuse of closed session
            self.session = None
            self._session_event_loop = None


async def main() -> int:
    """CLI entry point for fetching AniList data.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Fetch AniList data for an anime entry"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mal-id", type=int, help="MyAnimeList ID to fetch")
    group.add_argument(
        "--url", help="Full AniList anime URL (e.g. https://anilist.co/anime/21)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory path (default: current directory)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    helper = AniListHelper()
    ids = {"anilist_url": args.url}
    try:
        if args.url:
            canonical = await helper.fetch_all(ids, {}, args.output)
        else:
            anime_raw = await helper.fetch_anime_by_mal_id(args.mal_id)
            if not anime_raw:
                logger.error("No data found for the given MAL ID.")
                return 1
            anilist_id = anime_raw.get("id")
            if not anilist_id:
                logger.error("Could not resolve AniList ID from MAL ID.")
                return 1
            anilist_url = f"{_ANILIST_BASE}/{anilist_id}"
            ids = {"anilist_url": anilist_url}
            canonical = await helper.fetch_all(ids, {}, args.output)

        if canonical:
            logger.info(
                f"Data saved to {args.output}/ (anilist.jsonl, anilist_characters.jsonl)"
            )
            return 0
        else:
            logger.error("No data found for the given ID.")
            return 1
    except Exception:
        logger.exception("Error")
        return 1
    finally:
        await helper.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
