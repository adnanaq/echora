"""AniDB raw page fetcher implementing the PageFetcher interface."""

import asyncio
import gzip
import re
import time
from datetime import timedelta
from typing import Any, ClassVar

import aiohttp
import defusedxml.ElementTree as ET
from common.config import EnrichmentConfig
from enrichment.sources.anidb.anidb_character_crawler import (
    _BASE_URL as _CHAR_BASE_URL,
)
from enrichment.sources.anidb.anidb_character_crawler import (
    _CLEARANCE_SITE,
    _fetch_page_html,
    _has_character_data,
    _is_antileech,
    _solve_antileech,
)
from enrichment.sources.anidb.anidb_helper import _state
from enrichment.sources.base.browser import browser_session
from enrichment.sources.base.cloudflare_clearance import save_clearance
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceParseError,
)
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_cache_headers,
    build_html_response,
    validate_page_key,
)
from http_cache import HTTPCacheManager
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse

_ANIDB_API_URL = "http://api.anidb.net:9001/httpapi"
_XML_DECLARATION_PATTERN = re.compile(r"^\s*(?:<\?xml[^?]*\?>\s*)*", re.IGNORECASE)


def _decode_anidb_payload(raw_bytes: bytes) -> str:
    """Decompress gzipped AniDB API bytes when needed and decode UTF-8 text.

    Args:
        raw_bytes: Raw HTTP response bytes from AniDB XML API.

    Returns:
        Decoded XML text string.
    """
    if raw_bytes.startswith(b"\x1f\x8b"):
        raw_bytes = gzip.decompress(raw_bytes)
    return raw_bytes.decode("utf-8", errors="replace")


def _build_anidb_headers(
    max_age: timedelta | None,
    *,
    cache_only: bool,
) -> dict[str, str]:
    """Build Cache-Control headers for an AniDB HTTP request.

    Args:
        max_age: Optional cache freshness bound.
        cache_only: When True, add ``only-if-cached`` so no network call leaves
            the process during a ban cooldown.

    Returns:
        Request headers mapping.
    """
    headers = build_cache_headers(max_age)
    if not cache_only:
        return headers
    existing_control = headers.get("Cache-Control")
    headers["Cache-Control"] = (
        f"{existing_control}, only-if-cached" if existing_control else "only-if-cached"
    )
    return headers


def _extract_anidb_error(raw_text: str) -> tuple[str, str] | None:
    """Inspect leading XML tag and return ``(code, message)`` for ``<error>`` roots.

    Skips DOM parsing for normal multi-megabyte ``<anime>`` payloads and parses
    only compact ``<error>`` responses.

    Args:
        raw_text: Decoded response text from the AniDB HTTP XML API.

    Returns:
        Tuple of error code and lowercased error message, or None for normal
        anime XML payloads.

    Raises:
        ServiceParseError: If the payload is not XML or has malformed error XML.
    """
    body_start = _XML_DECLARATION_PATTERN.sub("", raw_text, count=1)
    if not body_start.startswith("<"):
        cause_message = "AniDB response is not valid XML"
        raise ServiceParseError(service="anidb", cause=cause_message)
    if not body_start.lower().startswith("<error"):
        return None
    try:
        root = ET.fromstring(raw_text)
    except ET.ParseError as exc:
        raise ServiceParseError(service="anidb", cause=exc) from exc
    if root.tag.lower() != "error":
        return None
    error_code = (root.get("code") or "").strip()
    error_message = (root.text or "").strip().lower()
    return error_code, error_message


class AniDBFetcher(PageFetcher):
    """Fetches raw AniDB XML (anime) and HTML (character) page responses."""

    provider: ClassVar[str] = "anidb"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME, EntityType.ENTITY_TYPE_CHARACTER}
    )

    def __init__(self, config: EnrichmentConfig, cache: HTTPCacheManager) -> None:
        """Initialize the AniDB fetcher.

        Args:
            config: Enrichment configuration carrying AniDB credentials and limits.
            cache: HTTP cache manager supplying the aiohttp session.
        """
        self._config = config
        self._cache = cache
        self._cache.network_request_pacer.set_minimum_seconds_between_requests(
            "anidb", config.anidb_min_request_interval
        )

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw AniDB responses for an anime or character page key.

        Args:
            page_key: AniDB PageKey with a numeric address.
            max_age: Optional cache freshness bound; zero forces a live fetch.

        Returns:
            FetchedPage holding raw XML/HTML PageResponse objects or not-found status.
        """
        validate_page_key(
            page_key,
            self.provider,
            self.supported_entity_types,
            numeric_address=True,
        )
        if page_key.entity_type == EntityType.ENTITY_TYPE_ANIME:
            return await self._fetch_anime_page(page_key, max_age=max_age)
        return await self._fetch_character_page(page_key)

    async def _fetch_anime_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None,
    ) -> FetchedPage:
        """Fetch raw AniDB HTTP XML API response for one anime ID.

        Args:
            page_key: AniDB anime PageKey.
            max_age: Optional cache freshness bound.

        Returns:
            FetchedPage holding the raw XML PageResponse or not-found status.
        """
        ban_left = max(0.0, _state.ban_lifts_at - time.time())
        cache_only = False
        if ban_left > 0:
            if not self._cache.enabled or (
                max_age is not None and max_age <= timedelta(0)
            ):
                ban_message = f"AniDB client banned for another {ban_left:.0f}s"
                raise ServiceBlockedError(ban_message, service="anidb")
            cache_only = True

        params: dict[str, Any] = {
            "request": "anime",
            "aid": int(page_key.address),
            "client": self._config.anidb_client,
            "clientver": self._config.anidb_clientver,
            "protover": self._config.anidb_protover,
        }
        headers = _build_anidb_headers(max_age, cache_only=cache_only)
        async with _state.lock():
            raw_text = await self._execute_xml_get(
                params, headers=headers, cache_only=cache_only
            )

        anidb_error = _extract_anidb_error(raw_text)
        if anidb_error is not None:
            error_code, error_message = anidb_error
            if error_code == "555" or "banned" in error_message:
                _state.ban_lifts_at = time.time() + self._config.anidb_ban_cooldown
                ban_message = "AniDB returned client ban response"
                raise ServiceBlockedError(ban_message, service="anidb")
            if "anime not found" in error_message:
                return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
            cause_message = f"AniDB error response: {raw_text[:200]}"
            raise ServiceNetworkError(service="anidb", cause=cause_message)

        return FetchedPage(
            page_key=page_key,
            responses=(
                PageResponse(
                    request_name="main",
                    content_type="application/xml",
                    body=raw_text.encode("utf-8"),
                ),
            ),
        )

    async def _execute_xml_get(
        self,
        params: dict[str, Any],
        *,
        headers: dict[str, str],
        cache_only: bool = False,
    ) -> str:
        """Send one AniDB XML API GET request and return decoded XML text.

        Args:
            params: Query parameters for AniDB HTTP API.
            headers: Cache-Control request headers.
            cache_only: When True, map HTTP 504 cache misses to ServiceBlockedError.

        Returns:
            Decoded XML text string.
        """
        status = 200
        try:
            async with self._cache.get_aiohttp_session(
                "anidb", timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(
                    _ANIDB_API_URL, params=params, headers=headers
                ) as response:
                    status = response.status
                    if (cache_only and status == 504) or status == 555:
                        raw_bytes = b""
                    else:
                        response.raise_for_status()
                        raw_bytes = await response.read()
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            raise ServiceNetworkError(service="anidb", cause=exc) from exc
        if cache_only and status == 504:
            miss_message = "AniDB client is banned and page is not cached"
            raise ServiceBlockedError(miss_message, service="anidb")
        if status == 555:
            _state.ban_lifts_at = time.time() + self._config.anidb_ban_cooldown
            ban_message = "AniDB returned HTTP 555 ban"
            raise ServiceBlockedError(ban_message, service="anidb")
        return _decode_anidb_payload(raw_bytes)

    async def _fetch_character_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw AniDB character HTML response via browser session.

        Args:
            page_key: AniDB character PageKey.

        Returns:
            FetchedPage holding the raw HTML PageResponse or not-found status.
        """
        url = f"{_CHAR_BASE_URL}/{page_key.address}"
        async with browser_session(
            headless=False, clearance_site=_CLEARANCE_SITE
        ) as session:
            read, current_page = await _fetch_page_html(session.browser, url)
            if read is None:
                await session.restart()
                await asyncio.sleep(2)
                read, current_page = await _fetch_page_html(session.browser, url)
            if read is None or read.html is None:
                cause_message = f"failed to read AniDB character page {url}"
                raise ServiceNetworkError(service="anidb", cause=cause_message)
            html = read.html
            if _is_antileech(html):
                if not await _solve_antileech(current_page):
                    blocked_message = f"antileech did not clear for {url}"
                    raise ServiceBlockedError(blocked_message, service="anidb")
                await current_page
                html = await current_page.get_content()
            if not _has_character_data(html):
                return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
            if read.challenged:
                await save_clearance(session.browser, _CLEARANCE_SITE)
        return FetchedPage(
            page_key=page_key,
            responses=(build_html_response("main", html),),
        )
