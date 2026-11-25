"""Kitsu raw page fetcher implementing the PageFetcher interface."""

import asyncio
import json
import logging
from contextlib import nullcontext
from datetime import timedelta
from typing import Any, ClassVar, cast

import aiohttp
from common.utils.retry import retry_with_backoff
from enrichment.sources.base.exceptions import ServiceNotFoundError
from enrichment.sources.base.network_errors import is_aiohttp_network_error
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_cache_headers,
    read_response_bytes,
    validate_page_key,
)
from http_cache import HTTPCacheManager
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse

logger = logging.getLogger(__name__)

_PAGE_SIZE = 20
_INTER_PAGE_SLEEP_S = 0.1
_VOICE_CONCURRENCY = 5


class KitsuFetcher(PageFetcher):
    """Fetches raw Kitsu JSON:API responses for anime and character pages."""

    provider: ClassVar[str] = "kitsu"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME, EntityType.ENTITY_TYPE_CHARACTER}
    )

    def __init__(self, cache: HTTPCacheManager) -> None:
        """Initialize the Kitsu fetcher.

        Args:
            cache: HTTP cache manager supplying aiohttp sessions.
        """
        self.base_url = "https://kitsu.io/api/edge"
        self._cache = cache

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw JSON:API responses for a Kitsu anime or character page.

        Args:
            page_key: Kitsu PageKey with a numeric or slug address.
            max_age: Optional cache freshness bound; zero forces a live fetch.

        Returns:
            FetchedPage holding raw PageResponse objects or not-found status.
        """
        validate_page_key(page_key, self.provider, self.supported_entity_types)
        async with self._cache.get_aiohttp_session(
            "kitsu", timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            if page_key.entity_type == EntityType.ENTITY_TYPE_ANIME:
                return await self._fetch_anime_page(
                    page_key, max_age=max_age, session=session
                )
            return await self._fetch_character_page(
                page_key, max_age=max_age, session=session
            )

    async def _resolve_resource_id(
        self,
        collection_endpoint: str,
        address: str,
        *,
        max_age: timedelta | None,
        session: Any,
    ) -> str | None:
        """Resolve a numeric ID or slug address into Kitsu's numeric resource ID.

        Args:
            collection_endpoint: Kitsu collection endpoint (e.g., ``"/anime"``).
            address: Kitsu numeric ID or slug string.
            max_age: Optional cache freshness bound.
            session: Active aiohttp session.

        Returns:
            Numeric Kitsu resource ID string, or None if slug lookup matched nothing.
        """
        if address.isdigit():
            return address
        try:
            _, _, payload, _ = await self._make_raw_request(
                collection_endpoint,
                params={"filter[slug]": address},
                max_age=max_age,
                session=session,
            )
        except ServiceNotFoundError:
            return None
        items = payload.get("data") or []
        if not items:
            return None
        return str(items[0]["id"])

    async def _fetch_anime_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None,
        session: Any,
    ) -> FetchedPage:
        """Fetch raw Kitsu anime, classification, episode, and character list pages.

        Args:
            page_key: Kitsu anime PageKey.
            max_age: Optional cache freshness bound.
            session: Active aiohttp session.

        Returns:
            FetchedPage holding raw responses or not-found status.
        """
        anime_id = await self._resolve_resource_id(
            "/anime", page_key.address, max_age=max_age, session=session
        )
        if anime_id is None:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        try:
            main_raw, main_type, _, _ = await self._make_raw_request(
                f"/anime/{anime_id}", max_age=max_age, session=session
            )
        except ServiceNotFoundError:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        sub_endpoints: tuple[tuple[str, str, dict[str, Any] | None], ...] = (
            (f"/anime/{anime_id}/genres", "genres_page_", None),
            (f"/anime/{anime_id}/categories", "categories_page_", None),
            (
                f"/anime/{anime_id}/anime-productions",
                "productions_page_",
                {"include": "producer"},
            ),
            (f"/anime/{anime_id}/mappings", "mappings_page_", None),
            (f"/anime/{anime_id}/episodes", "episodes_page_", None),
            (
                f"/anime/{anime_id}/characters",
                "characters_page_",
                {"include": "character"},
            ),
        )
        sub_results = await asyncio.gather(
            *[
                self._paginate_raw(
                    endpoint, prefix, params=params, max_age=max_age, session=session
                )
                for endpoint, prefix, params in sub_endpoints
            ]
        )
        responses = [
            PageResponse(request_name="main", content_type=main_type, body=main_raw)
        ]
        for page_group in sub_results:
            responses.extend(page_group)
        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _fetch_character_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None,
        session: Any,
    ) -> FetchedPage:
        """Fetch raw Kitsu character profile, media-characters, and voices pages.

        Args:
            page_key: Kitsu character PageKey.
            max_age: Optional cache freshness bound.
            session: Active aiohttp session.

        Returns:
            FetchedPage holding raw responses or not-found status.
        """
        char_id = await self._resolve_resource_id(
            "/characters", page_key.address, max_age=max_age, session=session
        )
        if char_id is None:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
        try:
            main_raw, main_type, _, _ = await self._make_raw_request(
                f"/characters/{char_id}", max_age=max_age, session=session
            )
        except ServiceNotFoundError:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        media_char_pages = await self._paginate_raw(
            f"/characters/{char_id}/media-characters",
            "media_characters_page_",
            params={"include": "media"},
            max_age=max_age,
            session=session,
        )
        media_char_ids: list[str] = []
        for page_response in media_char_pages:
            decoded = json.loads(page_response.body.decode("utf-8"))
            media_char_ids.extend(item["id"] for item in decoded.get("data", []))

        sem = asyncio.Semaphore(_VOICE_CONCURRENCY)

        async def _fetch_voices(mc_id: str) -> list[PageResponse]:
            async with sem:
                return await self._paginate_raw(
                    f"/media-characters/{mc_id}/voices",
                    f"voices_{mc_id}_page_",
                    params={"include": "person"},
                    max_age=max_age,
                    session=session,
                )

        voice_groups = await asyncio.gather(
            *[_fetch_voices(mc_id) for mc_id in media_char_ids]
        )
        responses = [
            PageResponse(request_name="main", content_type=main_type, body=main_raw),
            *media_char_pages,
        ]
        for group in voice_groups:
            responses.extend(group)
        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _paginate_raw(
        self,
        endpoint: str,
        request_prefix: str,
        *,
        params: dict[str, Any] | None = None,
        max_age: timedelta | None = None,
        session: Any | None = None,
    ) -> list[PageResponse]:
        """Paginate a Kitsu collection endpoint and return raw PageResponse objects.

        Args:
            endpoint: Kitsu API collection path.
            request_prefix: Prefix for generated ``request_name`` values.
            params: Optional extra query parameters.
            max_age: Optional cache freshness bound.
            session: Optional aiohttp session to reuse.

        Returns:
            Ordered list of raw PageResponse objects across all pages.
        """
        responses: list[PageResponse] = []
        page = 0
        collected_count = 0
        base_params = dict(params or {})
        while True:
            page_params = {
                **base_params,
                "page[limit]": _PAGE_SIZE,
                "page[offset]": page * _PAGE_SIZE,
            }
            raw_bytes, content_type, payload, from_cache = await self._make_raw_request(
                endpoint, page_params, max_age=max_age, session=session
            )
            items = payload.get("data")
            if not items:
                if page == 0:
                    responses.append(
                        PageResponse(
                            request_name=f"{request_prefix}1",
                            content_type=content_type,
                            body=raw_bytes,
                        )
                    )
                break
            responses.append(
                PageResponse(
                    request_name=f"{request_prefix}{page + 1}",
                    content_type=content_type,
                    body=raw_bytes,
                )
            )
            collected_count += len(items)
            total_count = payload.get("meta", {}).get("count", 0)
            if collected_count >= total_count or len(items) < _PAGE_SIZE:
                break
            page += 1
            if not from_cache:
                await asyncio.sleep(_INTER_PAGE_SLEEP_S)
        return responses

    async def _make_raw_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        max_age: timedelta | None = None,
        session: Any | None = None,
    ) -> tuple[bytes, str, dict[str, Any], bool]:
        """Perform a GET request to Kitsu and return raw bytes plus decoded JSON.

        Args:
            endpoint: Kitsu API path appended to ``base_url``.
            params: Optional query parameters.
            max_age: Optional cache freshness bound.
            session: Optional aiohttp session to reuse.

        Returns:
            Tuple of raw bytes, content type, decoded JSON, and cache flag.
        """
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
            **build_cache_headers(max_age),
        }
        url = f"{self.base_url}{endpoint}"

        async def _execute() -> tuple[bytes, str, dict[str, Any], bool]:
            ctx = (
                nullcontext(session)
                if session is not None
                else self._cache.get_aiohttp_session(
                    "kitsu", timeout=aiohttp.ClientTimeout(total=30)
                )
            )
            async with ctx as active_session:
                async with cast(aiohttp.ClientSession, active_session).get(
                    url, headers=headers, params=params
                ) as response:
                    if response.status == 404:
                        raise ServiceNotFoundError(  # noqa: TRY003
                            f"not found: {endpoint}", service="kitsu"
                        )
                    if response.status != 200:
                        response.raise_for_status()
                    raw_bytes, content_type = await read_response_bytes(response)
                    decoded = cast(
                        dict[str, Any], json.loads(raw_bytes.decode("utf-8"))
                    )
                    from_cache = bool(getattr(response, "from_cache", False))
                    return raw_bytes, content_type, decoded, from_cache

        return await retry_with_backoff(
            operation=_execute,
            max_retries=3,
            retry_delay=1.0,
            is_transient_error=is_aiohttp_network_error,
        )
