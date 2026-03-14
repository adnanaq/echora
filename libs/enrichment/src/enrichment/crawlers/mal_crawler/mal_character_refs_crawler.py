"""MAL Character Refs Crawler.

Fetches the characters list page for an anime:
    fetch_mal_character_refs(anime_id, anime_url)  — /anime/{id}/characters → list[CharacterRef]

A single fetch returns ALL character references (id, name, role, favorites).
Full character detail (bio, VAs, animeography) requires separate calls via mal_character_crawler.
"""

import json
import logging
import re
from typing import Any

from enrichment.crawlers.crawl4ai_docker import crawl_single_url
from enrichment.crawlers.mal_crawler.mal_base import (
    MAL_BASE_URL,
    get_mal_docker_browser_config,
    get_mal_docker_crawler_config,
    get_mal_scraping_limiter,
)
from enrichment.crawlers.mal_crawler.mal_models import CharacterRef
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = get_mal_scraping_limiter()


def _get_characters_schema() -> dict[str, Any]:
    """CSS schema to grab the raw characters table HTML for post-processing."""
    return {
        "name": "MalCharactersList",
        "baseSelector": "body",
        "fields": [
            {
                "name": "characters_html",
                "selector": "div#content table",
                "type": "html",
            }
        ],
    }


def _parse_character_refs(characters_html: str) -> list[CharacterRef]:
    """Parse the characters list HTML into CharacterRef objects.

    MAL structure: each character is a table block within
    table.js-anime-character-table. We identify characters by their
    /character/{id} links.

    Args:
        characters_html: Raw HTML of the characters list table.

    Returns:
        List of CharacterRef objects with IDs, roles, and favorites.
    """
    refs = []
    seen_ids: set[int] = set()

    char_blocks = re.split(
        r'<table[^>]*class="[^"]*js-anime-character-table[^"]*"[^>]*>', characters_html
    )

    for block in char_blocks[1:]:
        char_link = re.search(
            r'<a[^>]*href="[^"]*myanimelist[^"]*/character/(\d+)/([^"]*)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not char_link:
            continue

        char_id = int(char_link.group(1))
        if char_id in seen_ids:
            continue
        seen_ids.add(char_id)

        char_name = re.sub(r"<[^>]+>", "", char_link.group(3)).strip()

        role = "Supporting"
        role_match = re.search(
            r'<div[^>]*class="[^"]*js-chara-roll-and-name[^"]*"[^>]*>([^<]*)</div>',
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not role_match:
            role_text_match = re.search(r"\b(Main|Supporting)\b", block)
            if role_text_match:
                role = role_text_match.group(1)
        else:
            role_text = role_match.group(1).strip()
            if role_text.startswith("m_"):
                role = "Main"
            elif role_text.startswith("s_"):
                role = "Supporting"
            else:
                role = role_text or "Supporting"

        fav_match = re.search(
            r'<div[^>]*class="[^"]*js-anime-character-favorites[^"]*"[^>]*>\s*(\d+)',
            block,
            re.DOTALL | re.IGNORECASE,
        )
        favorites = int(fav_match.group(1)) if fav_match else 0

        refs.append(
            CharacterRef(
                char_id=char_id,
                name=char_name,
                role=role,
                favorites=favorites,
            )
        )

    return refs


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_character_ids",
    dependencies=[_get_characters_schema, _parse_character_refs],
)
async def _fetch_mal_characters_data(anime_id: int, anime_url: str) -> dict[str, Any] | None:
    """Fetch /anime/{id}/characters and extract character references. Cached by anime_id + anime_url."""
    url = f"{anime_url}/characters"

    await _limiter.acquire()
    result = await crawl_single_url(
        url=url,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(
            _get_characters_schema(),
            strategy_type="JsonCssExtractionStrategy",
            wait_until="networkidle",
            delay=2.0,
            magic=True,
        ),
    )
    if not result:
        logger.error(f"No result for characters page of anime {anime_id}")
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for characters page of anime {anime_id}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    if not raw_list:
        return None
    html = raw_list[0].get("characters_html") or ""
    if not html:
        return None
    return {"characters_html": html}


async def fetch_mal_character_refs(anime_id: int, anime_url: str) -> list[CharacterRef]:
    """Fetch all character references from /anime/{id}/characters.

    A single page fetch returns ALL characters (e.g., 1475 for One Piece).
    Role and favorites are included. Full character detail (bio, VAs, animeography)
    requires separate calls via fetch_mal_characters().

    Args:
        anime_id: MAL anime ID.
        anime_url: Canonical anime URL (e.g. https://myanimelist.net/anime/57334/Dandadan).
            Required to derive the slug-based characters URL and avoid redirect-related
            CSS selector misses.

    Returns:
        List of CharacterRef objects, empty on failure.
    """
    logger.info(f"[characters] Fetching character list for anime {anime_id}...")
    data = await _fetch_mal_characters_data(anime_id, anime_url)
    if not data:
        return []

    characters_html = data.get("characters_html", "")
    if not characters_html:
        logger.warning(f"No characters HTML extracted for anime {anime_id}")
        return []

    return _parse_character_refs(characters_html)
