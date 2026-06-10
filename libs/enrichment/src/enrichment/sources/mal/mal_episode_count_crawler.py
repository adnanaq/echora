"""MAL episode count crawler — zendriver + lxml XPath.

Fetches the aired episode count for any anime:
    fetch_mal_episode_count(anime_url)  — reads episode list page → int

MAL renders a span "(12/12)" or "(1,155/Unknown)" next to the Episodes
heading. The first number (before the slash) is always the current aired
count. This works for all anime regardless of status.
"""

import logging
import re
from typing import Any, cast

from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_EPISODE_COUNT_XPATH = (
    "//h2[@class='h2_overwrite'][text()='Episodes']/following-sibling::span[1]"
)


def _extract_episode_count(html: str) -> str | None:
    """Extract the episode counter span text from the episode list page.

    Args:
        html: Full HTML of the MAL episode list page.

    Returns:
        Raw counter string e.g. ``"(1,155/Unknown)"`` or ``"(12/12)"``,
        or None if the span is not found.
    """
    if not html:
        return None
    tree = etree.fromstring(html, etree.HTMLParser(encoding="utf-8"))
    els = cast(list[Any], tree.xpath(_EPISODE_COUNT_XPATH))
    if not els:
        return None
    return "".join(els[0].itertext()).strip()


async def _fetch_episode_count_html(url: str) -> str | None:
    """Navigate to a MAL episode list page and return its HTML.

    Args:
        url: Full MAL episode list URL
            (e.g. ``https://myanimelist.net/anime/21/One_Piece/episode``).

    Returns:
        Rendered page HTML, or None on failure.
    """
    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        page = await browser.get(url)
        await page.wait_for(selector="h2.h2_overwrite", timeout=15)
        return await page.get_content()
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None
    finally:
        try:
            await browser.stop()
        except Exception:
            pass


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_episode_count",
    dependencies=[_extract_episode_count],
)
async def _fetch_episode_count_data(url: str) -> str | None:
    """Fetch the episode list page and return the episode counter span text.

    Args:
        url: Full MAL episode list URL.

    Returns:
        Raw counter string e.g. ``"(1,155/Unknown)"``, or None on failure.
    """
    html = await _fetch_episode_count_html(url)
    if not html:
        logger.error(f"No result for episode list page {url}")
        return None
    counter = _extract_episode_count(html)
    if not counter:
        logger.error(f"Episode count span not found on {url}")
        return None
    return counter


async def fetch_mal_episode_count(anime_url: str) -> int:
    """Return the current aired episode count for an anime.

    Reads the episode counter span on the episode list page, e.g.:
      ``"(12/12)"``        → 12   (finished anime)
      ``"(1,155/Unknown)"`` → 1155 (ongoing anime)

    Returns 0 on failure or when no episodes are listed.

    Args:
        anime_url: Full MAL anime slug URL
            (e.g. ``https://myanimelist.net/anime/21/One_Piece``).

    Returns:
        Current aired episode count, or 0 on failure.
    """
    counter = await _fetch_episode_count_data(f"{anime_url}/episode")
    if not counter:
        return 0
    m = re.search(r"\(([0-9,]+)/", counter)
    return int(m.group(1).replace(",", "")) if m else 0
