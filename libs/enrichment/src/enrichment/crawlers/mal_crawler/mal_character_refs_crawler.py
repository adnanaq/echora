"""MAL Character Refs Crawler.

Fetches the characters list page for an anime:
    fetch_mal_character_refs(url)  — /anime/{id}/characters → list[str]

A single fetch returns ALL character URLs.
Full character detail (bio, VAs, animeography) requires separate calls via mal_character_crawler.
"""

import json
import logging
from typing import Any

from enrichment.crawlers.crawl4ai_docker import crawl_single_url
from enrichment.crawlers.mal_crawler.mal_base import (
    get_mal_docker_browser_config,
    get_mal_docker_crawler_config,
    get_mal_scraping_limiter,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = get_mal_scraping_limiter()


def _get_characters_schema() -> dict[str, Any]:
    """XPath schema to extract character URLs from the MAL characters page.

    Each character is contained in a table.js-anime-character-table.
    The first cell contains the character link with a /character/{id}/{slug} href.
    """
    return {
        "name": "MalCharactersList",
        "baseSelector": "//table[contains(@class, 'js-anime-character-table')]",
        "fields": [
            {
                "name": "url",
                "selector": ".//td[1]//a[contains(@href, '/character/')]",
                "type": "attribute",
                "attribute": "href",
            }
        ],
    }


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_character_ids",
    dependencies=[_get_characters_schema],
)
async def _fetch_mal_characters_data(url: str) -> list[dict[str, Any]] | None:
    """Fetch /anime/{id}/characters and extract character URLs. Cached by url."""
    await _limiter.acquire()
    result = await crawl_single_url(
        url=url,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(
            _get_characters_schema(),
            strategy_type="JsonXPathExtractionStrategy",
            wait_until="networkidle",
            delay=2.0,
            magic=False,
        ),
    )
    if not result:
        logger.error(f"No result for characters page {url}")
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for characters page {url}")
        return None

    items = json.loads(result.get("extracted_content") or "[]")
    return items if items else None


async def fetch_mal_character_refs(url: str) -> list[str]:
    """Fetch all character URLs from a MAL characters page.

    A single fetch returns ALL character URLs (e.g., 1475 for One Piece).
    Full character detail (bio, VAs, animeography) requires separate calls
    via fetch_mal_characters().

    Args:
        url: Full characters page URL
            (e.g. https://myanimelist.net/anime/57334/Dandadan/characters).

    Returns:
        Deduplicated list of character URLs, empty on failure.
    """
    logger.info(f"[characters] Fetching character list from {url}...")
    items = await _fetch_mal_characters_data(url)
    if not items:
        logger.warning(f"No character URLs extracted from {url}")
        return []

    return list(dict.fromkeys(  # deduplicate, preserve order
        item["url"] for item in items if item.get("url")
    ))
