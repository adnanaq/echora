"""AniSearch Character Refs Crawler.

Fetches the characters list page for an anime:
    fetch_anisearch_character_refs(anime_identifier)  →  list[dict]

Each dict contains {"url": str, "role": str}. All other character data
(name, description, favorites, VAs, ography) is extracted by the detail crawler.
"""

import logging
from typing import cast

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

_XPATHS: dict[str, str] = {
    section_id: f"//section[@id='{section_id}']//a[contains(@href,'character/')]/@href"
    for section_id in _SECTION_ROLE_MAP
}


def _extract_refs_from_html(html_text: str) -> dict[str, list[str]] | None:
    """Parse a /characters page into a {section_id: [href, ...]} dict."""
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    return {
        section_id: cast(list[str], tree.xpath(_XPATHS[section_id]))
        for section_id in _SECTION_ROLE_MAP
    }


def _absolutize(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"{_ANISEARCH_BASE_URL}/{href.lstrip('/')}"


def _post_process_refs(raw: dict[str, list[str]]) -> list[dict[str, str]]:
    """Flatten per-section hrefs into a deduplicated list of {url, role} dicts."""
    seen: set[str] = set()
    refs: list[dict[str, str]] = []
    for section_id, role_label in _SECTION_ROLE_MAP.items():
        for href in raw.get(section_id) or []:
            href = href.strip()
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
    dependencies=[_extract_refs_from_html],
)
async def _fetch_anisearch_character_refs_data(
    characters_url: str,
) -> list[dict[str, str]] | None:
    """Fetch /anime/{id},{slug}/characters and extract character refs. Cached by URL."""
    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        try:
            page = await browser.get(characters_url)
            await page.wait_for(selector="#content", timeout=10)
            await page.scroll_down(amount=1000, speed=3000)
            html_text = await page.get_content()
        except Exception as exc:
            logger.exception(f"navigation failed for {characters_url}")
            return None
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")

    if not html_text:
        logger.error(f"No HTML from characters page {characters_url}")
        return None

    raw = _extract_refs_from_html(html_text)
    if raw is None:
        logger.error(f"Failed to parse characters page {characters_url}")
        return None

    refs = _post_process_refs(raw)
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
    logger.info("Fetching AniSearch character list from %s...", characters_url)
    refs = await _fetch_anisearch_character_refs_data(characters_url)
    if not refs:
        logger.warning("No character refs extracted from %s", characters_url)
        return []
    return refs
