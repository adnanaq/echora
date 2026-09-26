"""MAL character refs crawler — zendriver + lxml XPath.

Fetches the /anime/{id}/characters list page and returns all character URLs:
    fetch_mal_character_refs(url)  — characters page URL → list[str]

A single fetch returns ALL character URLs (e.g. 1476 for One Piece).
Full character detail (bio, VAs, animeography) requires separate calls
via mal_character_crawler.
"""

import asyncio
import logging
from typing import Any, cast

from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_CHARACTER_TABLE_XPATH = "//table[contains(@class,'js-anime-character-table')]"
_CHARACTER_LINK_IN_TABLE_XPATH = ".//td[1]//a[contains(@href,'/character/')]/@href"


def _extract_character_urls(html: str) -> list[str]:
    """Return every character URL on a cast list page, in page order.

    MAL renders each character as its own table. Evaluating the link expression
    once per table, rather than once over the whole page, selects exactly the
    same links: a single ``//table//td[1]//a`` expression over the page took 60 s
    on One Piece's 1,481 tables, growing faster than the number of tables, where
    the per-table form takes 0.1 s. Extraction runs on the event loop, so the
    slow form stalled every other crawl for that minute.

    Args:
        html: A rendered MAL ``/characters`` page.

    Returns:
        Unique character URLs in the order they appear.
    """
    if not html:
        return []
    try:
        tree = etree.fromstring(html.encode(), etree.HTMLParser(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse character refs HTML")
        return []
    tables = cast(list[Any], tree.xpath(_CHARACTER_TABLE_XPATH))
    urls = [
        link
        for table in tables
        for link in cast(list[str], table.xpath(_CHARACTER_LINK_IN_TABLE_XPATH))
    ]
    return list(dict.fromkeys(urls))


async def _fetch_characters_page_html(url: str) -> str | None:
    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        page = await browser.get(url)
        await page.wait_for(selector="table.js-anime-character-table", timeout=30)
        await asyncio.sleep(2)
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
    ttl=TTL_MAL,
    key_prefix="mal_character_ids",
    dependencies=[_extract_character_urls],
)
async def _fetch_mal_characters_data(url: str) -> list[str] | None:
    """Fetch /anime/{id}/characters and extract character URLs. Cached by url."""
    html = await _fetch_characters_page_html(url)
    if not html:
        logger.error(f"No result for characters page {url}")
        return None
    urls = _extract_character_urls(html)
    if not urls:
        logger.error(f"No character URLs extracted from {url}")
        return None
    return urls


async def fetch_mal_character_refs(url: str) -> list[str]:
    """Fetch all character URLs from a MAL characters page.

    A single fetch returns ALL character URLs (e.g., 1476 for One Piece).
    Full character detail (bio, VAs, animeography) requires separate calls
    via fetch_mal_characters().

    Args:
        url: Full characters page URL
            (e.g. https://myanimelist.net/anime/57334/Dandadan/characters).

    Returns:
        Deduplicated list of character URLs, empty on failure.
    """
    logger.info(f"Fetching MAL character list from {url}...")
    urls = await _fetch_mal_characters_data(url)
    if not urls:
        logger.warning(f"No character URLs extracted from {url}")
        return []
    return urls
