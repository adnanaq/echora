#!/usr/bin/env python3
"""
AniList Helper for AI Enrichment Integration

Test script to fetch and analyze AniList data for anime entries using GraphQL API.
"""

import argparse
import asyncio
import json
import logging
import sys
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

import aiohttp

from http_cache.instance import http_cache_manager

logger = logging.getLogger(__name__)


class AniListEnrichmentHelper:
    """Helper for AniList data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """
        Create an AniListEnrichmentHelper and initialize its internal state.
        
        Attributes:
            base_url (str): AniList GraphQL endpoint URL.
            session (Optional[aiohttp.ClientSession]): Per-event-loop HTTP session, created lazily.
            rate_limit_remaining (int): Estimated remaining requests before hitting rate limit.
            rate_limit_reset (Optional[int]): Timestamp when the rate limit resets, if known.
            _session_event_loop (Optional[asyncio.AbstractEventLoop]): Event loop associated with the current session.
        """
        self.base_url = "https://graphql.anilist.co"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 90
        self.rate_limit_reset: Optional[int] = None
        self._session_event_loop: Optional[asyncio.AbstractEventLoop] = None

    async def _make_request(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a GraphQL request to the AniList API and return the parsed response data with cache metadata.

        Handles per-event-loop session management and respects server rate limits; on 429 responses it waits according to Retry-After and retries the request up to a maximum of 3 attempts.

        Parameters:
            query (str): GraphQL query string.
            variables (Optional[Dict[str, Any]]): Mapping of variables for the GraphQL query.

        Returns:
            result (Dict[str, Any]): The GraphQL `data` object merged into a dict. Contains an additional
            key `_from_cache` set to `True` if the response was served from cache, `False` otherwise.

        Raises:
            RuntimeError: If an aiohttp session cannot be initialized for the current event loop.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"query": query, "variables": variables or {}}

        # Check if we need to create/recreate session for current event loop
        current_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        if self.session is None or self._session_event_loop != current_loop:
            # Close old session if it exists
            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    logger.debug("Ignoring error while closing old session", exc_info=True)

            # Get cached session from centralized cache manager
            # Each event loop gets its own session via the cache manager
            # This enables GraphQL caching while preventing event loop conflicts
            self.session = http_cache_manager.get_aiohttp_session(
                "anilist",
                timeout=aiohttp.ClientTimeout(total=None),
                headers={
                    "X-Hishel-Body-Key": "true"
                },  # Enable body-based caching for GraphQL
            )
            logger.debug(
                "AniList cached session created via cache manager for current event loop"
            )
            self._session_event_loop = current_loop

        # Session should always be initialized by this point
        if self.session is None:
            raise RuntimeError("Failed to initialize AniList session")

        # Retry loop with bounded attempts to avoid unbounded recursion
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.rate_limit_remaining < 5:
                    logger.info(
                        f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds..."
                    )
                    await asyncio.sleep(60)
                    self.rate_limit_remaining = 90

                async with self.session.post(
                    self.base_url, json=payload, headers=headers
                ) as response:
                    # Capture cache status before response is consumed
                    from_cache = getattr(response, "from_cache", False)

                    if "X-RateLimit-Remaining" in response.headers:
                        self.rate_limit_remaining = int(
                            response.headers["X-RateLimit-Remaining"]
                        )

                    if response.status == 429:
                        if attempt < max_retries - 1:
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(
                                f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Waiting {retry_after} seconds..."
                            )
                            await asyncio.sleep(retry_after)
                            continue  # Retry in loop instead of recursion
                        else:
                            logger.error(
                                f"Rate limit exceeded after {max_retries} attempts. Giving up."
                            )
                            return {"_from_cache": from_cache}

                    response.raise_for_status()
                    data: Any = await response.json()
                    if "errors" in data:
                        logger.error(f"AniList GraphQL errors: {data['errors']}")
                        return {"_from_cache": from_cache}
                    result: Dict[str, Any] = data.get("data", {})
                    # Add cache metadata to result
                    result["_from_cache"] = from_cache
                    return result
            except (
                aiohttp.ClientError,
                aiohttp.ClientResponseError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
            ):
                # Network/JSON errors: log and return empty result
                logger.exception("AniList API request failed")
                return {"_from_cache": False}

        # Should not reach here, but defensive fallback
        return {"_from_cache": False}

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
        hashtag
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
        meanScore
        popularity
        favourites
        trending
        genres
        synonyms
        tags {
          id
          name
          description
          category
          rank
          isGeneralSpoiler
          isMediaSpoiler
          isAdult
        }
        relations {
          edges {
            node {
              id
              title {
                romaji
                english
              }
              format
              status
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
          color
          icon
        }
        streamingEpisodes {
          title
          thumbnail
          url
          site
        }
        nextAiringEpisode {
          episode
          airingAt
          timeUntilAiring
        }
        rankings {
          id
          rank
          type
          format
          year
          season
          allTime
          context
        }
        stats {
          scoreDistribution {
            score
            amount
          }
          statusDistribution {
            status
            amount
          }
        }
        updatedAt
        """

    # def _build_query_by_mal_id(self) -> str:
    #     return f"query ($idMal: Int) {{ Media(idMal: $idMal, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    def _build_query_by_anilist_id(self) -> str:
        return f"query ($id: Int) {{ Media(id: $id, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    # async def fetch_anime_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
    #     query = self._build_query_by_mal_id()
    #     variables = {"idMal": mal_id}
    #     response = await self._make_request(query, variables)
    #     return response.get("Media")

    async def fetch_anime_by_anilist_id(
        self, anilist_id: int
    ) -> Optional[Dict[str, Any]]:
        query = self._build_query_by_anilist_id()
        variables = {"id": anilist_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def _fetch_paginated_data(
        self, anilist_id: int, query_template: str, data_key: str
    ) -> List[Dict[str, Any]]:
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

            # Only rate limit for network requests, not cache hits
            # Cache hits are instant, no need to throttle
            if not response.get("_from_cache", False):
                await asyncio.sleep(0.5)

        return all_items

    async def fetch_all_characters(self, anilist_id: int) -> List[Dict[str, Any]]:
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
            characters(page: $page, perPage: 25, sort: ROLE) {
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
                  favourites
                  gender
                }
                role
                voiceActors(language: JAPANESE) {
                  id
                  name { full native }
                }
              }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "characters")

    async def fetch_all_staff(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            staff(page: $page, perPage: 25, sort: RELEVANCE) {
              pageInfo { hasNextPage }
              edges { node { id name { full native } } role }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "staff")

    async def fetch_all_episodes(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            airingSchedule(page: $page, perPage: 50) {
              pageInfo { hasNextPage }
              edges { node { id episode airingAt } }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "airingSchedule")

    async def _fetch_and_populate_details(
        self, anime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        anilist_id = anime_data.get("id")
        if not anilist_id:
            return anime_data

        details = {
            "characters": await self.fetch_all_characters(anilist_id),
            "staff": await self.fetch_all_staff(anilist_id),
            "airingSchedule": await self.fetch_all_episodes(anilist_id),
        }

        for key, data in details.items():
            if data:
                anime_data[key] = {"edges": data}
                logger.info(f"Total {key} fetched: {len(data)}")

        return anime_data

    # async def fetch_all_data_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
    #     anime_data = await self.fetch_anime_by_mal_id(mal_id)
    #     if not anime_data:
    #         logger.warning(f"No AniList data found for MAL ID: {mal_id}")
    #         return None
    #     return await self._fetch_and_populate_details(anime_data)

    async def fetch_all_data_by_anilist_id(
        self, anilist_id: int
    ) -> Optional[Dict[str, Any]]:
        anime_data = await self.fetch_anime_by_anilist_id(anilist_id)
        if not anime_data:
            logger.warning(f"No AniList data found for AniList ID: {anilist_id}")
            return None
        return await self._fetch_and_populate_details(anime_data)

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
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
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
    
    Parses command-line arguments --anilist-id (required) and optional --output (defaults to "test_anilist_output.json"), invokes the AniListEnrichmentHelper to retrieve and enrich anime data, and writes the JSON output when data is found. Ensures the helper is closed before exit.
    
    Returns:
        int: 0 when data was successfully fetched and saved; 1 when no data was found or an error occurred.
    """
    parser = argparse.ArgumentParser(description="Test AniList data fetching")
    group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--mal-id", type=int, help="MyAnimeList ID to fetch")
    group.add_argument("--anilist-id", type=int, help="AniList ID to fetch")
    parser.add_argument(
        "--output",
        type=str,
        default="test_anilist_output.json",
        help="Output file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    helper = AniListEnrichmentHelper()
    anime_data = None
    try:
        if args.anilist_id:
            try:
                anime_data = await helper.fetch_all_data_by_anilist_id(args.anilist_id)
            except Exception:
                logger.exception(
                    f"Error fetching AniList data for ID {args.anilist_id}"
                )
                anime_data = None
        # elif args.mal_id:
        #     anime_data = await helper.fetch_all_data_by_mal_id(args.mal_id)

        if anime_data:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {args.output}")
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