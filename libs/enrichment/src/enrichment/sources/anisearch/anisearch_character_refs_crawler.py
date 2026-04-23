"""AniSearch Character Refs Crawler.

Fetches the characters list page for an anime:
    fetch_anisearch_character_refs(anime_identifier)  →  list[dict]

Each dict contains {"url": str, "role": str}. All other character data
(name, description, favorites, VAs, ography) is extracted by the detail crawler.
"""

import json
import logging
from typing import Any

from enrichment.sources.base.crawl4ai_docker import crawl_single_url
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

BASE_ANIME_URL = "https://www.anisearch.com/anime/"
_ANISEARCH_BASE_URL = "https://www.anisearch.com"

# Section ID → role label (from h2 text observed on the live page).
# chara50 = role not yet classified by the community.
_SECTION_ROLE_MAP: dict[str, str] = {
    "chara1": "Main Character",
    "chara2": "Secondary Character",
    "chara3": "Extra",
    "chara4": "Organisation",
    "chara5": "Other",
    "chara50": "Unknown",
}


def _get_character_refs_schema() -> dict[str, Any]:
    """XPath schema — extract character hrefs from each role section separately.

    Per-section extraction ensures role attribution is structural (section ID),
    not inferred from adjacent h2 text which could vary by locale.
    The page is statically rendered, so domcontentloaded (docker default) suffices.
    """
    fields = [
        {
            "name": section_id,
            "selector": f"//section[@id='{section_id}']//a[contains(@href,'character/')]",
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
        for section_id in _SECTION_ROLE_MAP
    ]
    return {"name": "AniSearchCharacterRefs", "baseSelector": "//body", "fields": fields}


def _absolutize(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"{_ANISEARCH_BASE_URL}/{href.lstrip('/')}"


def _post_process_refs(raw: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten per-section results into a deduplicated list of {url, role} dicts."""
    seen: set[str] = set()
    refs: list[dict[str, str]] = []
    for section_id, role_label in _SECTION_ROLE_MAP.items():
        for item in raw.get(section_id) or []:
            href = (item.get("url") or "").strip()
            if not href:
                continue
            url = _absolutize(href)
            if url not in seen:
                seen.add(url)
                refs.append({"url": url, "role": role_label})
    return refs


def _normalize_characters_page_url(anime_identifier: str) -> str:
    """Normalize an anime identifier into a full AniSearch characters page URL."""
    if (
        anime_identifier.startswith(BASE_ANIME_URL)
        and "/characters" in anime_identifier
    ):
        return anime_identifier
    if anime_identifier.startswith(BASE_ANIME_URL):
        return f"{anime_identifier.rstrip('/')}/characters"
    clean_id = anime_identifier.lstrip("/").replace("/characters", "").rstrip("/")
    return f"{BASE_ANIME_URL}{clean_id}/characters"


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_character_refs",
    dependencies=[_get_character_refs_schema],
)
async def _fetch_anisearch_character_refs_data(
    characters_url: str,
) -> list[dict[str, str]] | None:
    """Fetch /anime/{id},{slug}/characters and extract character refs. Cached by URL."""
    result = await crawl_single_url(
        url=characters_url,
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(_get_character_refs_schema()),
    )
    if not result:
        logger.error(f"No result for characters page {characters_url}")
        return None

    status = result.get("status_code")
    if status and status >= 400:
        logger.error(f"HTTP {status} for characters page {characters_url}")
        return None
    if status and 300 <= status < 400:
        logger.debug(f"HTTP {status} (redirect) for characters page {characters_url}")

    items: list[dict[str, Any]] = json.loads(result.get("extracted_content") or "[]")
    if not items:
        logger.warning(f"Empty extracted content from {characters_url}")
        return None

    refs = _post_process_refs(items[0])
    return refs or None


async def fetch_anisearch_character_refs(
    anime_identifier: str,
) -> list[dict[str, str]]:
    """Fetch all character refs from an AniSearch anime characters page.

    Args:
        anime_identifier: Full URL, path, or canonical ID
            (e.g. "18878,dan-da-dan" or "https://www.anisearch.com/anime/18878,dan-da-dan").

    Returns:
        List of {"url": str, "role": str} dicts. Empty list on failure.
    """
    characters_url = _normalize_characters_page_url(anime_identifier)
    logger.info(f"Fetching AniSearch character list from {characters_url}...")
    refs = await _fetch_anisearch_character_refs_data(characters_url)
    if not refs:
        logger.warning(f"No character refs extracted from {characters_url}")
        return []
    return refs
