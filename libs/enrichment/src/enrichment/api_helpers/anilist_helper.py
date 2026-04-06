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
from common.utils.retry import retry_with_backoff
from http_cache.instance import http_cache_manager

from ..exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)
from ..mappers.anilist_mapper import anime_from_anilist, character_from_anilist
from .anilist.anilist_anime_models import AniListAnime
from .anilist.anilist_character_models import AniListCharacterEdge

logger = logging.getLogger(__name__)


class AniListEnrichmentHelper:
    """Helper for AniList data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        self.base_url = "https://graphql.anilist.co"
        self.session: aiohttp.ClientSession | None = None
        self.rate_limit_remaining = 90
        self._session_event_loop: asyncio.AbstractEventLoop | None = None

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

        headers = {"Content-Type": "application/json", "Accept": "application/json", "X-Hishel-Body-Key": "true"}
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
        """
        GraphQL selection set for requesting Media (ANIME) fields from the AniList API.

        Returns:
            str: A multiline GraphQL field selection string that requests comprehensive Media fields
                 (titles, identifiers, images, stats, relations, studios, external links, tags,
                 rankings, airing/trailer info, and other metadata) used when querying an AniList
                 Media node.
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
        query = self._build_query_by_mal_id()
        variables = {"idMal": mal_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def fetch_anime(self, anilist_id: int) -> dict[str, Any] | None:
        query = self._build_query_by_anilist_id()
        variables = {"id": anilist_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def _fetch_paginated_data(
        self, anilist_id: int, query_template: str, data_key: str
    ) -> list[dict[str, Any]]:
        """
        Fetches and accumulative list of paginated edges for a specific Media sub-resource by AniList ID.

        Parameters:
            anilist_id (int): AniList numeric identifier for the Media node to query.
            query_template (str): GraphQL query string that accepts `id` and `page` variables and returns a pagination structure under `Media`.
            data_key (str): Key name under `Media` whose pagination container provides `edges` and `pageInfo` (e.g., "characters", "staff", "airingSchedule").

        Returns:
            List[Dict[str, Any]]: Concatenated list of edge objects from all fetched pages. Paging stops when the container is missing, empty, or `pageInfo.hasNextPage` is false. If a response indicates it was served from cache via a `_from_cache` flag, the routine does not apply the inter-page throttle delay.
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
        """
        Fetches all character edges for an anime from AniList by its AniList ID.

        Parameters:
                anilist_id (int): AniList numeric ID of the anime to query.

        Returns:
                characters (List[Dict[str, Any]]): A list of edge dictionaries from the GraphQL `characters` connection; each edge contains `node` (character data), `role`, and `voiceActors`.
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
        self, anilist_id: int, output_dir: str | None = None
    ) -> dict[str, Any] | None:
        """
        Fetch anime data, validate via AniListAnime, map to canonical fields.

        Writes the raw API response to {output_dir}/anilist.json when output_dir
        is provided (consistent with the MAL/AnimSchedule pattern).

        Returns:
            Canonical dict (output of anime_from_anilist), or None if not found.
        """
        raw = await self.fetch_anime(anilist_id)
        if not raw:
            logger.warning(f"No AniList data found for AniList ID: {anilist_id}")
            return None

        anime = AniListAnime.model_validate(raw)
        canonical = anime_from_anilist(anime)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "anilist.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(canonical, ensure_ascii=False) + "\n")
            logger.debug(f"Saved canonical AniList data to {out_path}")

        return canonical

    async def fetch_characters_canonical(
        self, anilist_id: int, output_dir: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Fetch all character edges, validate, map to canonical dicts.

        Writes raw edges as JSONL to {output_dir}/anilist_characters.jsonl when
        output_dir is provided.

        Returns:
            List of canonical character dicts (output of character_from_anilist).
        """
        raw_edges = await self.fetch_characters(anilist_id)
        if not raw_edges:
            return []

        canonical = []
        for edge in raw_edges:
            try:
                char_edge = AniListCharacterEdge.model_validate(edge)
                canonical.append(character_from_anilist(char_edge))
            except Exception:
                logger.warning("Skipping invalid character edge", exc_info=True)

        if output_dir and canonical:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "anilist_characters.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for char in canonical:
                    f.write(json.dumps(char, ensure_ascii=False) + "\n")
            logger.debug(f"Saved {len(canonical)} canonical characters to {out_path}")

        return canonical

    async def fetch_all(
        self, anilist_id: int, output_dir: str | None = None
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Fetch canonical anime data and all characters for an AniList ID.

        Args:
            anilist_id: AniList anime ID.
            output_dir: If provided, writes anilist.jsonl and anilist_characters.jsonl
                to this directory (consistent with MAL/AnimSchedule pattern).

        Returns:
            Tuple of (canonical anime dict or None, list of canonical character dicts).
        """
        logger.info(f"Fetching AniList data for: {anilist_id}")
        anime = await self.fetch_anime_canonical(anilist_id, output_dir)
        if anime:
            logger.info(f"AniList anime fetched: {anime.get('title', anilist_id)}")
        characters = await self.fetch_characters_canonical(anilist_id, output_dir)
        logger.info(f"AniList characters fetched: {len(characters)} characters")
        return anime, characters

    async def fetch_all_data_by_mal_id(self, mal_id: int) -> dict[str, Any] | None:
        anime_data = await self.fetch_anime_by_mal_id(mal_id)
        if not anime_data:
            logger.warning(f"No AniList data found for MAL ID: {mal_id}")
            return None
        characters = await self.fetch_characters(anime_data["id"])
        if characters:
            anime_data["characters"] = {"edges": characters}
        return anime_data

    async def close(self) -> None:
        """
        Close the helper's active aiohttp session if one exists.

        If a session is present, closes the underlying ClientSession and resets session state to make the helper safe for potential reuse.
        """
        if self.session:
            await self.session.close()
            # Reset session state to prevent accidental reuse of closed session
            self.session = None
            self._session_event_loop = None

    async def __aenter__(self) -> "AniListEnrichmentHelper":
        """
        Enter the async context manager for this helper.

        The underlying HTTP session is not created on enter; it will be created lazily on first request.

        Returns:
            AniListEnrichmentHelper: The helper instance (`self`).
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """
        Exit the asynchronous context and close the helper's HTTP session.

        Returns:
            False to indicate exceptions are not suppressed.
        """
        await self.close()
        return False


async def main() -> int:
    """
    CLI entry point that fetches AniList data for a provided ID and writes the result as JSON to a file.

    Parses command-line arguments --anilist-id or --mal-id (mutually exclusive, one required) and optional --output (defaults to "."), invokes the AniListEnrichmentHelper to retrieve and enrich anime data, and writes the JSON output when data is found. Ensures the helper is closed before exit.

    Returns:
        int: 0 when data was successfully fetched and saved; 1 when no data was found or an error occurred.
    """
    parser = argparse.ArgumentParser(
        description="Fetch AniList data for an anime entry"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mal-id", type=int, help="MyAnimeList ID to fetch")
    group.add_argument("--anilist-id", type=int, help="AniList ID to fetch")
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory path (default: current directory)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    helper = AniListEnrichmentHelper()
    try:
        if args.anilist_id:
            canonical = await helper.fetch_anime_canonical(args.anilist_id, args.output)
            await helper.fetch_characters_canonical(args.anilist_id, args.output)
        else:
            anime_raw = await helper.fetch_anime_by_mal_id(args.mal_id)
            if not anime_raw:
                logger.error("No data found for the given MAL ID.")
                return 1
            anilist_id = anime_raw.get("id")
            if not anilist_id:
                logger.error("Could not resolve AniList ID from MAL ID.")
                return 1
            canonical = await helper.fetch_anime_canonical(anilist_id, args.output)
            await helper.fetch_characters_canonical(anilist_id, args.output)

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
