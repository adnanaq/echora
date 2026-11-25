"""Anime-Planet raw page fetcher implementing the PageFetcher interface."""

import logging
from datetime import timedelta
from typing import Any, ClassVar
from urllib.parse import urlsplit

from enrichment.sources.base.browser import browser_session
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_html_response,
    validate_page_key,
)
from enrichment.sources.base.page_readiness import wait_for_page
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey

logger = logging.getLogger(__name__)

_AP_BASE_URL = "https://www.anime-planet.com"
_ANIME_PLANET_SITE = "Anime-Planet"
_SEARCH_PAGE_PATH = "/search.php"
_ANIME_SELECTOR = "section.entryBar, div.quickFilter"
_CHAR_REFS_SELECTOR = "a.name[href*='/characters/']"
_CHARACTER_SELECTOR = "h1[itemprop='name'], div.quickFilter"


async def _fetch_ap_page(
    browser: Any,
    url: str,
    selector: str,
    *,
    required: bool = True,
) -> tuple[str, str] | None:
    """Navigate to an Anime-Planet URL, clear Cloudflare if needed, and return HTML.

    Args:
        browser: Active zendriver browser instance.
        url: Target Anime-Planet URL.
        selector: CSS readiness selector to wait for.
        required: When True, raise ServiceNetworkError on navigation failure.

    Returns:
        Tuple of HTML string and final URL, or None when optional and unavailable.
    """
    try:
        page = await browser.get(url)
        await wait_for_page(page, selector, url, cloudflare_site=_ANIME_PLANET_SITE)
        final_url = page.url or url
        html = await page.get_content()
    except Exception as exc:
        if required:
            cause_message = f"navigation failed for {url}: {exc}"
            raise ServiceNetworkError(
                service="anime_planet", cause=cause_message
            ) from exc
        logger.warning(f"Optional Anime-Planet page failed for {url}: {exc}")
        return None
    if not html:
        return None
    return html, final_url


class AnimePlanetFetcher(PageFetcher):
    """Fetches raw Anime-Planet HTML responses for anime and character pages."""

    provider: ClassVar[str] = "anime_planet"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME, EntityType.ENTITY_TYPE_CHARACTER}
    )

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw HTML responses for an Anime-Planet anime or character page key.

        Args:
            page_key: Anime-Planet PageKey with a slug address.
            max_age: Optional cache freshness bound (unused by live browser fetch).

        Returns:
            FetchedPage holding raw HTML PageResponse objects or not-found status.
        """
        del max_age
        validate_page_key(page_key, self.provider, self.supported_entity_types)
        if page_key.entity_type == EntityType.ENTITY_TYPE_ANIME:
            return await self._fetch_anime_page(page_key)
        return await self._fetch_character_page(page_key)

    async def _fetch_anime_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw Anime-Planet anime main and /characters HTML responses.

        Args:
            page_key: Anime-Planet anime PageKey.

        Returns:
            FetchedPage holding raw HTML responses or not-found status.
        """
        url = f"{_AP_BASE_URL}/anime/{page_key.address}"
        async with browser_session(headless=True) as session:
            main_result = await _fetch_ap_page(
                session.browser, url, _ANIME_SELECTOR, required=True
            )
            if main_result is None:
                return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
            main_html, final_url = main_result
            if urlsplit(final_url).path == _SEARCH_PAGE_PATH:
                return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

            responses = [build_html_response("main", main_html)]
            chars_result = await _fetch_ap_page(
                session.browser,
                f"{final_url.rstrip('/')}/characters",
                _CHAR_REFS_SELECTOR,
                required=False,
            )
            if (
                chars_result is not None
                and urlsplit(chars_result[1]).path != _SEARCH_PAGE_PATH
            ):
                responses.append(build_html_response("characters", chars_result[0]))
        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _fetch_character_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw Anime-Planet character detail HTML response.

        Args:
            page_key: Anime-Planet character PageKey.

        Returns:
            FetchedPage holding raw main HTML response or not-found status.
        """
        url = f"{_AP_BASE_URL}/characters/{page_key.address}"
        async with browser_session(headless=True) as session:
            result = await _fetch_ap_page(
                session.browser, url, _CHARACTER_SELECTOR, required=True
            )
        if result is None:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
        html, final_url = result
        if urlsplit(final_url).path == _SEARCH_PAGE_PATH:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)
        return FetchedPage(
            page_key=page_key, responses=(build_html_response("main", html),)
        )
