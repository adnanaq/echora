"""AniList raw page fetcher implementing the PageFetcher interface."""

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any, ClassVar

import aiohttp
from common.utils.retry import retry_with_backoff
from enrichment.sources.base.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceParseError,
    ServiceRateLimitedError,
)
from enrichment.sources.base.network_errors import is_aiohttp_network_error
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_cache_headers,
    read_response_bytes,
    validate_page_key,
)
from http_cache import HTTPCacheManager
from http_cache.aiohttp_adapter import CachedAiohttpSession
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse

logger = logging.getLogger(__name__)

_MAX_RATE_LIMIT_WAITS = 3
_RETRY_AFTER_DEFAULT_S = 60
_RATE_LIMIT_REMAINING_THRESHOLD = 5
_RATE_LIMIT_SLEEP_S = 60
_INTER_PAGE_SLEEP_S = 0.5

_MEDIA_FIELDS = """
id
idMal
title { romaji english native userPreferred }
description(asHtml: false)
source
format
episodes
duration
status
season
seasonYear
startDate { year month day }
endDate { year month day }
countryOfOrigin
isAdult
coverImage { extraLarge large medium color }
bannerImage
trailer { id site thumbnail }
averageScore
popularity
favourites
stats { scoreDistribution { score amount } }
genres
synonyms
tags { name description category isAdult }
relations {
  edges {
    node {
      id idMal
      title { romaji english }
      format status seasonYear averageScore
      coverImage { extraLarge }
      episodes chapters volumes
    }
    relationType
  }
}
studios { edges { node { id name isAnimationStudio } isMain } }
externalLinks { id url site type language }
nextAiringEpisode { episode airingAt timeUntilAiring }
rankings { rank context format year season allTime }
"""

_ANIME_QUERY = (
    f"query ($id: Int) {{ Media(id: $id, type: ANIME) {{ {_MEDIA_FIELDS} }} }}"
)

_ANIME_CHARACTERS_QUERY = """
query ($id: Int!, $page: Int!) {
  Media(id: $id, type: ANIME) {
    characters(page: $page, perPage: 50, sort: [ROLE, ID]) {
      pageInfo { hasNextPage }
      edges {
        node {
          id
          name { full native alternative alternativeSpoiler }
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
          voiceActor { id name { full native } languageV2 image { large } siteUrl }
          roleNotes
          dubGroup
        }
      }
    }
  }
}
"""

_CHARACTER_QUERY = """
query ($id: Int!, $page: Int!) {
  Character(id: $id) {
    id
    name { full native alternative alternativeSpoiler }
    image { large medium }
    description
    gender
    dateOfBirth { year month day }
    age
    bloodType
    favourites
    siteUrl
    media(page: $page, perPage: 25) {
      pageInfo { hasNextPage }
      edges {
        characterRole
        voiceActorRoles {
          voiceActor { id name { full native } languageV2 image { large } siteUrl }
          roleNotes
          dubGroup
        }
        node {
          id
          type
          title { romaji english native }
        }
      }
    }
  }
}
"""


def _is_transient_anilist_error(error: Exception) -> bool:
    """Return True for transient AniList errors eligible for retry.

    Args:
        error: Exception raised during an AniList request.

    Returns:
        True when the error is transient and safe to retry.
    """
    if isinstance(
        error, (ServiceRateLimitedError, ServiceBlockedError, AniListGraphQLError)
    ):
        return False
    if isinstance(error, ServiceNetworkError):
        return isinstance(
            error.cause, (aiohttp.ClientError, json.JSONDecodeError, TimeoutError)
        )
    return is_aiohttp_network_error(error)


class AniListFetcher(PageFetcher):
    """Fetches raw AniList GraphQL JSON responses for anime and character pages."""

    provider: ClassVar[str] = "anilist"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME, EntityType.ENTITY_TYPE_CHARACTER}
    )

    def __init__(self, cache: HTTPCacheManager) -> None:
        """Initialize the AniList fetcher.

        Args:
            cache: HTTP cache manager supplying the aiohttp session.
        """
        self.base_url = "https://graphql.anilist.co"
        self._cache = cache
        self.session: aiohttp.ClientSession | CachedAiohttpSession | None = None
        self.rate_limit_remaining = 90

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw GraphQL responses for an AniList anime or character page.

        Args:
            page_key: AniList PageKey with a numeric address.
            max_age: Optional cache freshness bound; zero forces a live fetch.

        Returns:
            FetchedPage holding raw PageResponse objects or not-found status.
        """
        validate_page_key(
            page_key,
            self.provider,
            self.supported_entity_types,
            numeric_address=True,
        )
        target_id = int(page_key.address)
        if page_key.entity_type == EntityType.ENTITY_TYPE_ANIME:
            return await self._fetch_anime_page(page_key, target_id, max_age=max_age)
        return await self._fetch_character_page(page_key, target_id, max_age=max_age)

    async def _fetch_anime_page(
        self,
        page_key: PageKey,
        anilist_id: int,
        *,
        max_age: timedelta | None,
    ) -> FetchedPage:
        """Fetch raw anime metadata and paginated character connection responses.

        Args:
            page_key: AniList anime PageKey.
            anilist_id: Numeric AniList anime ID.
            max_age: Optional cache freshness bound.

        Returns:
            FetchedPage holding raw responses or not-found status.
        """
        main_result = await self._make_raw_request(
            _ANIME_QUERY, {"id": anilist_id}, max_age=max_age
        )
        if main_result is None:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        main_bytes, main_type, main_payload, _ = main_result
        data = main_payload.get("data")
        if not isinstance(data, dict) or "Media" not in data:
            msg = "AniList anime response missing 'data.Media' field"
            raise ServiceParseError(service="anilist", cause=msg)
        if data["Media"] is None:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        responses = [
            PageResponse(request_name="main", content_type=main_type, body=main_bytes)
        ]
        character_pages = await self._fetch_paginated_raw(
            _ANIME_CHARACTERS_QUERY,
            anilist_id,
            resource_key="Media",
            connection_key="characters",
            request_prefix="characters_page_",
            max_age=max_age,
        )
        responses.extend(character_pages)
        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _fetch_character_page(
        self,
        page_key: PageKey,
        character_id: int,
        *,
        max_age: timedelta | None,
    ) -> FetchedPage:
        """Fetch raw character profile and paginated media appearance responses.

        Args:
            page_key: AniList character PageKey.
            character_id: Numeric AniList character ID.
            max_age: Optional cache freshness bound.

        Returns:
            FetchedPage holding raw responses or not-found status.
        """
        page = 1
        responses: list[PageResponse] = []
        while True:
            result = await self._make_raw_request(
                _CHARACTER_QUERY, {"id": character_id, "page": page}, max_age=max_age
            )
            if result is None:
                if page == 1:
                    return FetchedPage(
                        page_key=page_key, responses=(), is_not_found=True
                    )
                msg = f"AniList character pagination failed at page {page} with 404"
                raise ServiceNetworkError(service="anilist", cause=msg)
            raw_bytes, content_type, payload, from_cache = result
            data = payload.get("data")
            if not isinstance(data, dict) or "Character" not in data:
                msg = "AniList character response missing 'data.Character' field"
                raise ServiceParseError(service="anilist", cause=msg)
            character_data = data["Character"]
            if character_data is None:
                if page == 1:
                    return FetchedPage(
                        page_key=page_key, responses=(), is_not_found=True
                    )
                msg = f"AniList character pagination returned null at page {page}"
                raise ServiceParseError(service="anilist", cause=msg)
            request_name = "main" if page == 1 else f"media_page_{page}"
            responses.append(
                PageResponse(
                    request_name=request_name, content_type=content_type, body=raw_bytes
                )
            )
            has_next = (
                character_data.get("media", {})
                .get("pageInfo", {})
                .get("hasNextPage", False)
            )
            if not has_next:
                break
            page += 1
            if not from_cache:
                await asyncio.sleep(_INTER_PAGE_SLEEP_S)
        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _fetch_paginated_raw(
        self,
        query: str,
        target_id: int,
        *,
        resource_key: str,
        connection_key: str,
        request_prefix: str,
        max_age: timedelta | None,
    ) -> list[PageResponse]:
        """Fetch all pages of a GraphQL connection as raw PageResponse objects.

        Args:
            query: GraphQL query string accepting ``id`` and ``page`` variables.
            target_id: Numeric AniList entity ID.
            resource_key: Top-level field under ``data`` (e.g., ``"Media"``).
            connection_key: Paginated connection field name (e.g., ``"characters"``).
            request_prefix: Prefix for generated ``request_name`` values.
            max_age: Optional cache freshness bound.

        Returns:
            Ordered list of raw PageResponse objects across all pages.
        """
        page = 1
        responses: list[PageResponse] = []
        while True:
            result = await self._make_raw_request(
                query, {"id": target_id, "page": page}, max_age=max_age
            )
            if result is None:
                msg = f"AniList {connection_key} pagination failed at page {page} with 404"
                raise ServiceNetworkError(service="anilist", cause=msg)
            raw_bytes, content_type, payload, from_cache = result
            data = payload.get("data")
            resource = data.get(resource_key) if isinstance(data, dict) else None
            if not isinstance(resource, dict) or connection_key not in resource:
                msg = (
                    f"AniList pagination missing 'data.{resource_key}.{connection_key}'"
                )
                raise ServiceParseError(service="anilist", cause=msg)
            connection = resource[connection_key]
            if not connection:
                if page == 1:
                    break
                msg = f"AniList {connection_key} pagination empty at page {page}"
                raise ServiceParseError(service="anilist", cause=msg)
            responses.append(
                PageResponse(
                    request_name=f"{request_prefix}{page}",
                    content_type=content_type,
                    body=raw_bytes,
                )
            )
            has_next = connection.get("pageInfo", {}).get("hasNextPage", False)
            if not has_next:
                break
            page += 1
            if not from_cache:
                await asyncio.sleep(_INTER_PAGE_SLEEP_S)
        return responses

    async def _ensure_session(self) -> None:
        """Lazily create the aiohttp session on first use."""
        if self.session is None:
            self.session = self._cache.get_aiohttp_session(
                "anilist", timeout=aiohttp.ClientTimeout(total=None)
            )
        if self.session is None:
            error_message = "Failed to initialize AniList session"
            raise RuntimeError(error_message)

    async def _read_checked_response(
        self,
        response: Any,
    ) -> tuple[bytes, str, dict[str, Any], bool] | None:
        """Read and validate a non-429 AniList GraphQL HTTP response.

        Args:
            response: Open aiohttp client response.

        Returns:
            Tuple of raw bytes, content type, decoded JSON, and cache flag, or None on 404.
        """
        from_cache = bool(getattr(response, "from_cache", False))
        if response.status == 403:
            blocked_message = "API disabled/blocked"
            raise ServiceBlockedError(blocked_message, service="anilist")
        if response.status == 404:
            return None
        response.raise_for_status()
        raw_bytes, content_type = await read_response_bytes(response)
        decoded = json.loads(raw_bytes.decode("utf-8"))
        if "errors" in decoded:
            raise AniListGraphQLError(decoded["errors"])
        if (
            not from_cache
            and self.rate_limit_remaining < _RATE_LIMIT_REMAINING_THRESHOLD
        ):
            await asyncio.sleep(_RATE_LIMIT_SLEEP_S)
            self.rate_limit_remaining = 90
        return raw_bytes, content_type, decoded, from_cache

    async def _execute_raw_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        max_age: timedelta | None = None,
    ) -> tuple[bytes, str, dict[str, Any], bool] | None:
        """Send one GraphQL POST request and return raw bytes plus decoded JSON.

        Args:
            query: GraphQL query string.
            variables: Optional GraphQL variables mapping.
            max_age: Optional cache freshness bound.

        Returns:
            Tuple of raw bytes, content type, decoded JSON, and cache flag, or None on 404.
        """
        await self._ensure_session()
        assert self.session is not None  # noqa: S101
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Hishel-Body-Key": "true",
            **build_cache_headers(max_age),
        }
        body = {"query": query, "variables": variables or {}}

        for attempt in range(_MAX_RATE_LIMIT_WAITS):
            try:
                async with self.session.post(
                    self.base_url, json=body, headers=headers
                ) as response:
                    if "X-RateLimit-Remaining" in response.headers:
                        self.rate_limit_remaining = int(
                            response.headers["X-RateLimit-Remaining"]
                        )
                    if response.status == 429:
                        if attempt < _MAX_RATE_LIMIT_WAITS - 1:
                            retry_after = int(
                                response.headers.get(
                                    "Retry-After", _RETRY_AFTER_DEFAULT_S
                                )
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        raise ServiceRateLimitedError(
                            service="anilist", attempts=_MAX_RATE_LIMIT_WAITS
                        )
                    return await self._read_checked_response(response)
            except aiohttp.ClientResponseError as exc:
                if exc.status >= 500:
                    raise ServiceNetworkError(service="anilist", cause=exc) from exc
                raise
            except (aiohttp.ClientError, json.JSONDecodeError, TimeoutError) as exc:
                raise ServiceNetworkError(service="anilist", cause=exc) from exc
        loop_error = "AniList rate-limit loop exited unexpectedly"
        raise AssertionError(loop_error)  # pragma: no cover

    async def _make_raw_request(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        max_age: timedelta | None = None,
    ) -> tuple[bytes, str, dict[str, Any], bool] | None:
        """Send a GraphQL request with retry on transient network failures.

        Args:
            query: GraphQL query string.
            variables: Optional GraphQL variables mapping.
            max_age: Optional cache freshness bound.

        Returns:
            Tuple of raw bytes, content type, decoded JSON, and cache flag, or None on 404.
        """
        return await retry_with_backoff(
            operation=lambda: self._execute_raw_request(
                query, variables, max_age=max_age
            ),
            max_retries=3,
            retry_delay=1.0,
            is_transient_error=_is_transient_anilist_error,
        )

    async def close(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self.session:
            await self.session.close()
            self.session = None
