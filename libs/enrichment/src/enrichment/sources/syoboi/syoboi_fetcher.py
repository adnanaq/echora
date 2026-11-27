"""Syoboi Calendar raw page fetcher implementing the PageFetcher interface."""

from datetime import timedelta
from typing import ClassVar

import aiohttp
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_cache_headers,
    read_response_bytes,
    validate_page_key,
)
from http_cache import HTTPCacheManager
from lxml import etree
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse

_SYOBOI_DB_URL = "https://cal.syoboi.jp/db.php"


def _has_title_items(xml_bytes: bytes) -> bool:
    """Check whether a Syoboi TitleLookup XML response contains a TitleItem.

    Args:
        xml_bytes: Raw UTF-8 XML bytes from ``Command=TitleLookup``.

    Returns:
        True if at least one ``TitleItem`` element is present, False when valid
        XML holds no title items.

    Raises:
        ServiceParseError: If ``xml_bytes`` is malformed or truncated XML.
    """
    try:
        parser = etree.XMLParser(resolve_entities=False)
        root = etree.fromstring(xml_bytes, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ServiceParseError(service="syoboi", cause=exc) from exc
    return root.find(".//TitleItem") is not None


class SyoboiFetcher(PageFetcher):
    """Fetches raw Syoboi Calendar XML responses (TitleLookup and ProgLookup)."""

    provider: ClassVar[str] = "syoboi"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME}
    )

    def __init__(self, cache: HTTPCacheManager) -> None:
        """Initialize the Syoboi fetcher.

        Args:
            cache: HTTP cache manager supplying aiohttp sessions.
        """
        self._cache = cache

    async def _fetch_command(
        self,
        command: str,
        tid: str,
        *,
        max_age: timedelta | None,
    ) -> tuple[bytes, str]:
        """Fetch raw XML bytes and content type for one Syoboi db.php command.

        Args:
            command: Syoboi API command (``"TitleLookup"`` or ``"ProgLookup"``).
            tid: Numeric Syoboi title ID.
            max_age: Optional cache freshness bound.

        Returns:
            Tuple of raw XML bytes and MIME content type.
        """
        headers = build_cache_headers(max_age)
        params = {"Command": command, "TID": tid}
        try:
            async with self._cache.get_aiohttp_session(
                "syoboi", timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(
                    _SYOBOI_DB_URL, params=params, headers=headers
                ) as response:
                    response.raise_for_status()
                    return await read_response_bytes(response)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise ServiceNetworkError(service="syoboi", cause=exc) from exc

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw TitleLookup and ProgLookup XML responses for a Syoboi TID.

        Args:
            page_key: Syoboi anime PageKey with numeric TID address.
            max_age: Optional cache freshness bound; zero forces a live fetch.

        Returns:
            FetchedPage holding ``main`` and ``programs`` XML responses or not-found status.
        """
        validate_page_key(
            page_key,
            self.provider,
            self.supported_entity_types,
            numeric_address=True,
        )
        title_bytes, title_type = await self._fetch_command(
            "TitleLookup", page_key.address, max_age=max_age
        )
        if not _has_title_items(title_bytes):
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        prog_bytes, prog_type = await self._fetch_command(
            "ProgLookup", page_key.address, max_age=max_age
        )
        return FetchedPage(
            page_key=page_key,
            responses=(
                PageResponse(
                    request_name="main", content_type=title_type, body=title_bytes
                ),
                PageResponse(
                    request_name="programs", content_type=prog_type, body=prog_bytes
                ),
            ),
        )
