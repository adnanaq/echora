"""Anime-Planet Character Refs Crawler.

Fetches the characters list page for an anime:
    fetch_animeplanet_character_refs(url)  — /anime/{slug}/characters → list[dict]

Each dict contains {"url": "/characters/slug", "role": ""}.
Full character detail (bio, VAs, ography including per-title role) requires
separate calls via anime_planet_character_crawler.
"""

import json
import logging
from typing import Any

from enrichment.crawlers.crawl4ai_docker import crawl_single_url
from enrichment.crawlers.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_URL = "https://www.anime-planet.com"


def _get_characters_list_schema() -> dict[str, Any]:
    """XPath schema — extract character hrefs directly.

    Character links are server-rendered; domcontentloaded (the docker default)
    is sufficient. networkidle + delay caused 90s timeouts on large casts
    (e.g. One Piece: 1088 characters, ~1000 thumbnail image requests pending).
    """
    return {
        "name": "AnimePlanetCharactersList",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "characters",
                "selector": "//a[contains(@class,'name') and contains(@href,'/characters/')]",
                "type": "list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    }
                ],
            }
        ],
    }


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_character_refs",
    dependencies=[_get_characters_list_schema],
)
async def _fetch_refs_data(url: str) -> list[dict[str, str]] | None:
    """Fetch /anime/{slug}/characters and extract character hrefs. Cached by url."""
    result = await crawl_single_url(
        url=url,
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(_get_characters_list_schema()),
    )
    if not result:
        logger.error(f"No result for characters page {url}")
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for characters page {url}")
        return None

    items: list[dict[str, Any]] = json.loads(result.get("extracted_content") or "[]")
    if not items:
        logger.warning(f"Empty extracted content from {url}")
        return None

    characters: list[dict[str, str]] = items[0].get("characters") or []
    refs = [{"url": c["url"], "role": ""} for c in characters if c.get("url")]
    return refs or None


async def fetch_animeplanet_character_refs(url: str) -> list[dict[str, str]]:
    """Fetch all character refs from an Anime-Planet characters page.

    Args:
        url: Full characters page URL, e.g.
            https://www.anime-planet.com/anime/dandadan/characters

    Returns:
        List of {"url": "/characters/slug", "role": "Main|Secondary|Minor"} dicts.
        Empty list on failure.
    """
    logger.info(f"Fetching AP character list from {url}...")
    refs = await _fetch_refs_data(url)
    if not refs:
        logger.warning(f"No character refs extracted from {url}")
        return []
    return refs
