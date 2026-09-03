"""Anime-Planet Character Refs Crawler — zendriver + lxml XPath.

Fetches the characters list page for an anime:
    fetch_animeplanet_character_refs(url)  — /anime/{slug}/characters → list[dict]

Each dict contains {"url": "/characters/slug", "role": ""}.
Full character detail (bio, VAs, ography) requires separate calls via
anime_planet_character_crawler.
"""

import logging
from typing import Any, cast

from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_URL = "https://www.anime-planet.com"

_XPATHS: dict[str, str] = {
    "characters": "//a[contains(@class,'name') and contains(@href,'/characters/')]",
}


def _extract_refs_from_html(html: str) -> list[dict[str, str]] | None:
    """Extract character hrefs from a rendered AP characters list page.

    Args:
        html: Full rendered HTML of an Anime-Planet /anime/{slug}/characters page.

    Returns:
        List of ``{"url": "/characters/slug", "role": ""}`` dicts, or None if no
        character links were found.
    """
    if not html:
        return None
    tree = etree.fromstring(html, etree.HTMLParser(encoding="utf-8"))
    anchors = cast(list[Any], tree.xpath(_XPATHS["characters"]))
    hrefs = [el.get("href") for el in anchors if el.get("href")]
    if not hrefs:
        return None
    return [{"url": href, "role": ""} for href in hrefs]


async def _fetch_refs_html(url: str) -> str | None:
    """Fetch the characters list page HTML using zendriver.

    The page is server-rendered; domcontentloaded is sufficient. Using a
    presence-based wait instead of networkidle avoids 90-second timeouts on
    large casts (e.g. One Piece: 1088+ characters, ~1000 pending image
    requests).

    Args:
        url: Full Anime-Planet characters page URL.

    Returns:
        Rendered page HTML, or None on navigation failure.
    """
    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        page = await browser.get(url)
        await page.wait_for(selector="a.name[href*='/characters/']", timeout=20)
        return await page.get_content()
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_character_refs",
    dependencies=[_extract_refs_from_html],
)
async def _fetch_refs_data(url: str) -> list[dict[str, str]] | None:
    """Fetch /anime/{slug}/characters and extract character hrefs. Cached by url."""
    html = await _fetch_refs_html(url)
    if not html:
        logger.error(f"No HTML for characters page {url}")
        return None
    return _extract_refs_from_html(html)


async def fetch_animeplanet_character_refs(url: str) -> list[dict[str, str]]:
    """Fetch all character refs from an Anime-Planet characters page.

    Args:
        url: Full characters page URL, e.g.
            ``https://www.anime-planet.com/anime/dandadan/characters``.

    Returns:
        List of ``{"url": "/characters/slug", "role": ""}`` dicts.
        Empty list on failure.
    """
    logger.info(f"Fetching AP character list from {url}...")
    refs = await _fetch_refs_data(url)
    if not refs:
        logger.warning(f"No character refs extracted from {url}")
        return []
    return refs
