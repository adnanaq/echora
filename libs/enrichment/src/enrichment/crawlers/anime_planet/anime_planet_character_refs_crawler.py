"""Anime-Planet Character Refs Crawler.

Fetches the characters list page for an anime:
    fetch_animeplanet_character_refs(url)  — /anime/{slug}/characters → list[dict]

Each dict contains {"url": "/characters/slug", "role": "Main|Secondary|Minor"}.
Full character detail (bio, VAs, ography) requires separate calls via
anime_planet_character_crawler.
"""

import logging
import re
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

# Pattern: <h3 ... class="...sub...">Role text</h3> ... (content until next h3.sub or EOF)
_SECTION_RE = re.compile(
    r'<h3[^>]+class="[^"]*\bsub\b[^"]*"[^>]*>(.*?)</h3>(.*?)(?=<h3[^>]+class="[^"]*\bsub\b|$)',
    re.DOTALL | re.IGNORECASE,
)
# Matches: <a href="/characters/slug" class="name"> OR <a class="name" href="/characters/slug">
_CHAR_HREF_RE = re.compile(
    r'<a\s[^>]*href="(/characters/[^"?#]+)"[^>]*class="[^"]*\bname\b[^"]*"'
    r'|<a\s[^>]*class="[^"]*\bname\b[^"]*"[^>]*href="(/characters/[^"?#]+)"',
    re.IGNORECASE,
)


def _get_characters_list_schema() -> dict[str, Any]:
    """XPath schema — extract full body HTML for regex-based parsing."""
    return {
        "name": "AnimePlanetCharactersList",
        "baseSelector": "//body",
        "fields": [{"name": "page_html", "selector": "//body", "type": "html"}],
    }


def _role_from_header(raw_header: str) -> str:
    """Extract role name from section header text.

    'Main Characters' → 'Main', 'Secondary Characters' → 'Secondary', etc.
    Falls back to the full stripped text if the first word is not a known role.
    """
    text = re.sub(r"<[^>]+>", "", raw_header).strip()
    first_word = text.split()[0] if text else ""
    return first_word if first_word in {"Main", "Secondary", "Minor"} else text


def _parse_character_refs(page_html: str) -> list[dict[str, str]]:
    """Parse character refs from full body HTML.

    Returns a list of {"url": "/characters/slug", "role": "Main|..."} dicts,
    preserving document order within each section.
    """
    results: list[dict[str, str]] = []
    for section_match in _SECTION_RE.finditer(page_html):
        role = _role_from_header(section_match.group(1))
        section_content = section_match.group(2)
        for m in _CHAR_HREF_RE.finditer(section_content):
            url = m.group(1) or m.group(2)
            if url:
                results.append({"url": url, "role": role})
    return results


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_character_refs",
    dependencies=[_get_characters_list_schema],
)
async def _fetch_refs_data(url: str) -> list[dict[str, str]] | None:
    """Fetch /anime/{slug}/characters and extract character refs. Cached by url."""
    result = await crawl_single_url(
        url=url,
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(
            _get_characters_list_schema(),
            wait_until="networkidle",
            delay=2.0,
        ),
    )
    if not result:
        logger.error(f"No result for characters page {url}")
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for characters page {url}")
        return None

    page_html: str = result.get("html") or ""
    if not page_html:
        logger.warning(f"Empty page HTML from {url}")
        return None

    refs = _parse_character_refs(page_html)
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
