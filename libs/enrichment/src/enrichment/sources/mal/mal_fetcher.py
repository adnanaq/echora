"""MAL raw page fetcher implementing the PageFetcher interface."""

import asyncio
import logging
import re
from datetime import timedelta
from typing import Any, ClassVar

from enrichment.sources.base.browser import browser_session
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_html_response,
    extract_canonical_url,
    validate_page_key,
)
from enrichment.sources.base.page_readiness import wait_for_page
from enrichment.sources.mal.mal_base import MAL_DOMAIN
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey

logger = logging.getLogger(__name__)

_TITLE_SELECTOR = "h1.title-name"
_NOT_FOUND_SELECTOR = "div.error404"
_NOT_FOUND_MARKER = re.compile(r"""class=["'][^"']*\berror404\b""")
_GALLERY_SELECTOR = "div.border_solid"
_CHARACTERS_SELECTOR = "table.js-anime-character-table"
_CHARACTER_HEADER_SELECTOR = "h2.normal_header"
_INTER_REQUEST_DELAY = 3.0


async def _navigate_and_read(
    browser: Any,
    url: str,
    selector: str,
) -> tuple[str, str]:
    """Navigate to a MAL page, wait for a selector, and return HTML and final URL.

    Args:
        browser: Active zendriver browser instance.
        url: Target MAL URL.
        selector: CSS readiness selector to wait for.

    Returns:
        Tuple of rendered HTML and final URL.

    Raises:
        ServiceNetworkError: If navigation fails or yields empty HTML.
    """
    try:
        page = await browser.get(url)
        await wait_for_page(page, selector, url)
        html = await page.get_content()
        final_url = page.url or url
    except Exception as exc:
        cause_message = f"navigation failed for {url}: {exc}"
        raise ServiceNetworkError(service="mal", cause=cause_message) from exc

    if not html:
        empty_message = f"no HTML from {url}"
        raise ServiceNetworkError(service="mal", cause=empty_message)
    return html, final_url


async def _navigate_and_read_optional(
    browser: Any,
    url: str,
    selector: str,
) -> tuple[str, str] | None:
    """Navigate to an optional MAL sub-page, returning None on failure.

    Args:
        browser: Active zendriver browser instance.
        url: Target MAL URL.
        selector: CSS readiness selector to wait for.

    Returns:
        Tuple of rendered HTML and final URL, or None when unavailable.
    """
    try:
        return await _navigate_and_read(browser, url, selector)
    except ServiceNetworkError as exc:
        logger.warning(f"Optional MAL page failed for {url}: {exc}")
        return None


class MalFetcher(PageFetcher):
    """Fetches raw MyAnimeList HTML responses for anime and character pages."""

    provider: ClassVar[str] = "mal"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME, EntityType.ENTITY_TYPE_CHARACTER}
    )

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw HTML responses for a MAL anime or character page key.

        Args:
            page_key: MAL PageKey with a numeric address.
            max_age: Optional cache freshness bound (unused by live browser fetch).

        Returns:
            FetchedPage holding raw HTML PageResponse objects or not-found status.
        """
        del max_age
        validate_page_key(
            page_key,
            self.provider,
            self.supported_entity_types,
            numeric_address=True,
        )
        if page_key.entity_type == EntityType.ENTITY_TYPE_ANIME:
            return await self._fetch_anime_page(page_key)
        return await self._fetch_character_page(page_key)

    async def _fetch_anime_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw MAL anime main, /pics, and /characters HTML responses.

        Args:
            page_key: MAL anime PageKey.

        Returns:
            FetchedPage holding raw HTML responses or not-found status.
        """
        url = f"https://myanimelist.net/anime/{page_key.address}"
        async with browser_session(headless=True, allowed_site=MAL_DOMAIN) as session:
            browser = session.browser
            main_html, final_url = await _navigate_and_read(
                browser, url, f"{_TITLE_SELECTOR}, {_NOT_FOUND_SELECTOR}"
            )
            if _NOT_FOUND_MARKER.search(main_html):
                return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

            raw_base = extract_canonical_url(main_html) or final_url
            canonical_base = raw_base.split("?", maxsplit=1)[0].rstrip("/")
            responses = [build_html_response("main", main_html)]

            await asyncio.sleep(_INTER_REQUEST_DELAY)
            pics_result = await _navigate_and_read_optional(
                browser, f"{canonical_base}/pics", _GALLERY_SELECTOR
            )
            if pics_result is not None:
                responses.append(build_html_response("pics", pics_result[0]))

            await asyncio.sleep(_INTER_REQUEST_DELAY)
            chars_result = await _navigate_and_read_optional(
                browser,
                f"{canonical_base}/characters",
                _CHARACTERS_SELECTOR,
            )
            if chars_result is not None:
                responses.append(build_html_response("characters", chars_result[0]))

        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _fetch_character_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw MAL character detail HTML response.

        Args:
            page_key: MAL character PageKey.

        Returns:
            FetchedPage holding the main HTML response or not-found status.
        """
        url = f"https://myanimelist.net/character/{page_key.address}"
        async with browser_session(headless=True, allowed_site=MAL_DOMAIN) as session:
            html, _ = await _navigate_and_read(
                session.browser,
                url,
                f"{_CHARACTER_HEADER_SELECTOR}, {_NOT_FOUND_SELECTOR}",
            )
        if _NOT_FOUND_MARKER.search(html):
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
        return FetchedPage(
            page_key=page_key, responses=(build_html_response("main", html),)
        )
