"""AnimeSchedule raw page fetcher implementing the PageFetcher interface."""

from datetime import timedelta
from typing import ClassVar

import aiohttp
from enrichment.sources.animeschedule.animeschedule_helper import require_app_token
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_cache_headers,
    read_response_bytes,
    validate_page_key,
)
from http_cache import HTTPCacheManager
from pydantic import SecretStr
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse

_ANIME_DETAIL_BASE_URL = "https://animeschedule.net/api/v3/anime"


class AnimeScheduleFetcher(PageFetcher):
    """Fetches raw AnimeSchedule JSON responses for anime pages."""

    provider: ClassVar[str] = "animeschedule"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME}
    )

    def __init__(
        self,
        cache: HTTPCacheManager,
        app_token: SecretStr | None = None,
    ) -> None:
        """Initialize the AnimeSchedule fetcher.

        Args:
            cache: HTTP cache manager supplying aiohttp sessions.
            app_token: AnimeSchedule API bearer token.
        """
        self._cache = cache
        self._app_token = app_token

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw AnimeSchedule JSON response for a single anime slug.

        Args:
            page_key: AnimeSchedule anime PageKey with slug address.
            max_age: Optional cache freshness bound; zero forces a live fetch.

        Returns:
            FetchedPage holding raw JSON PageResponse or not-found status.
        """
        validate_page_key(page_key, self.provider, self.supported_entity_types)
        token = require_app_token(self._app_token).get_secret_value()
        headers = {
            "Authorization": f"Bearer {token}",
            **build_cache_headers(max_age),
        }
        url = f"{_ANIME_DETAIL_BASE_URL}/{page_key.address}"
        try:
            async with self._cache.get_aiohttp_session(
                "animeschedule", timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 404:
                        return FetchedPage(
                            page_key=page_key, responses=(), is_not_found=True
                        )
                    response.raise_for_status()
                    raw_bytes, content_type = await read_response_bytes(response)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise ServiceNetworkError(service="animeschedule", cause=exc) from exc

        return FetchedPage(
            page_key=page_key,
            responses=(
                PageResponse(
                    request_name="main", content_type=content_type, body=raw_bytes
                ),
            ),
        )
