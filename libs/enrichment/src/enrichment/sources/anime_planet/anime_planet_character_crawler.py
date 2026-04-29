"""Anime-Planet Character Detail Crawler.

Two public functions:
    fetch_animeplanet_character(url)   — single character detail page
    fetch_animeplanet_characters(urls) — batch character detail pages

All data (name, description, tags, alt names, voice actors, ography) is
extracted from the character detail page via XPath schema + Python regex helpers.
"""

import json
import logging
import re
from collections.abc import Callable
from html import unescape
from typing import Any

from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
    AnimePlanetCharacterAnimeRole,
    AnimePlanetCharacterMangaRole,
    AnimePlanetVoiceActor,
)
from enrichment.sources.anime_planet.animeplanet_mapper import (
    character_from_animeplanet,
)
from enrichment.sources.base.crawl4ai_docker import crawl_batch_urls
from enrichment.sources.base.crawler_config import (
    get_ap_rate_limiter,
    get_docker_browser_config,
    get_docker_crawler_config,
)
from enrichment.sources.base.framework import BaseCrawler, DockerTransport, NullRepository
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_URL = "https://www.anime-planet.com"

_CHARACTER_BATCH_SIZE = 20

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

_ENTRY_BAR_RE = re.compile(
    r'<section[^>]+class="[^"]*entryBar[^"]*"[^>]*>(.*?)</section>',
    re.DOTALL | re.IGNORECASE,
)
_GENDER_RE = re.compile(r"Gender:\s*([^\s<]+)", re.IGNORECASE)
_HAIR_RE = re.compile(r"Hair Color:\s*([^<\n]+?)(?:\s*</|$)", re.IGNORECASE)

_METADATA_ITEM_RE = re.compile(
    r"EntryMetadata__title[^>]*>\s*([^<]+?)\s*</h3>.*?EntryMetadata__value[^>]*>\s*([^<]+?)\s*</div>",
    re.DOTALL | re.IGNORECASE,
)

_ALT_NAMES_RE = re.compile(
    r'<h2[^>]+class="[^"]*\baka\b[^"]*"[^>]*>Aka:\s*(.*?)</h2>',
    re.DOTALL | re.IGNORECASE,
)

_DESCRIPTION_RE = re.compile(
    r'<div[^>]+itemprop="description">(.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)

_TAG_RE = re.compile(
    r'href="/characters/tags/[^"]+">([^<]+)</a>',
    re.IGNORECASE,
)

_ANIME_ROLES_SECTION_RE = re.compile(
    r"<h3>Anime Roles</h3>\s*<table[^>]*>(.*?)</tbody>\s*</table>",
    re.DOTALL | re.IGNORECASE,
)
_MANGA_ROLES_SECTION_RE = re.compile(
    r"<h3>Manga Roles</h3>\s*<table[^>]*>(.*?)</tbody>\s*</table>",
    re.DOTALL | re.IGNORECASE,
)

_TR_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)

_OGRAPHY_HREF_RE = re.compile(r'href="(/(?:anime|manga)/[^"?#]+)"', re.IGNORECASE)
_LAST_ANCHOR_TEXT_RE = re.compile(r">([^<>]+)</a>(?!.*</a>)", re.DOTALL | re.IGNORECASE)

# Voice actor flags: flagJP / flagUS / flagES / flagFR / flagDE / flagKO
_VA_FLAG_RE = re.compile(
    r'<div[^>]+class="flag\s+flag(JP|US|ES|FR|DE|KO)"[^>]*>.*?'
    r'<a[^>]+href="(/people/[^"?#]+)"[^>]*>([^<]+)</a>',
    re.DOTALL | re.IGNORECASE,
)

_FLAG_LANG_MAP: dict[str, str] = {
    "JP": "jp",
    "US": "us",
    "ES": "es",
    "FR": "fr",
    "DE": "de",
    "KO": "ko",
}


# ---------------------------------------------------------------------------
# XPath schema
# ---------------------------------------------------------------------------


def _get_character_schema() -> dict[str, Any]:
    """XPath schema — stable itemprop/href anchors + full body HTML for regex."""
    return {
        "name": "AnimePlanetCharacter",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "name",
                "selector": "//h1[@itemprop='name']",
                "type": "text",
            },
            {
                "name": "image",
                "selector": "//img[@itemprop='image']",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "loved_rank",
                "selector": "//a[contains(@href,'/characters/top-loved')]",
                "type": "text",
            },
            {
                "name": "hated_rank",
                "selector": "//a[contains(@href,'/characters/top-hated')]",
                "type": "text",
            },
            {
                "name": "loved_count",
                "selector": "//section[contains(@class,'sidebarStats')]//h3[contains(@class,'smSidebar')][.//span[@class='heartOn']]",
                "type": "text",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Regex helper functions
# ---------------------------------------------------------------------------


def _strip_tags(html: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", html)).strip()


def _parse_rank(raw: str | None) -> int | None:
    if not raw:
        return None
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else None


def _parse_loved_count(raw: str | None) -> int | None:
    """Parse love count from text like '36,485 users' → 36485."""
    if not raw:
        return None
    m = re.search(r"[\d,]+", raw)
    return int(m.group().replace(",", "")) if m else None


def _extract_entry_bar(body_html: str) -> dict[str, str | None]:
    """Extract gender and hair_color from the entryBar section HTML."""
    result: dict[str, str | None] = {"gender": None, "hair_color": None}
    section_match = _ENTRY_BAR_RE.search(body_html)
    if not section_match:
        return result
    bar_text = section_match.group(1)
    if m := _GENDER_RE.search(bar_text):
        result["gender"] = m.group(1).strip()
    if m := _HAIR_RE.search(bar_text):
        result["hair_color"] = m.group(1).strip()
    return result


def _extract_metadata(body_html: str) -> dict[str, str]:
    """Extract EntryMetadata title/value pairs as a flat dict."""
    return {
        m.group(1).strip(): m.group(2).strip()
        for m in _METADATA_ITEM_RE.finditer(body_html)
        if m.group(1).strip() and m.group(2).strip()
    }


def _extract_alt_names(body_html: str) -> list[str]:
    """Extract alternate names from the Aka: heading."""
    m = _ALT_NAMES_RE.search(body_html)
    if not m:
        return []
    raw = _strip_tags(m.group(1))
    return [n.strip() for n in raw.split(",") if n.strip()]


def _extract_description(body_html: str) -> str | None:
    """Extract plain-text description from the itemprop='description' div."""
    m = _DESCRIPTION_RE.search(body_html)
    if not m:
        return None
    text = _strip_tags(m.group(1))
    return text if text else None


def _extract_tags(body_html: str) -> list[str]:
    """Extract character tag names from /characters/tags/ anchor hrefs."""
    return [unescape(m.group(1).strip()) for m in _TAG_RE.finditer(body_html)]


def _extract_vas_from_cell(cell_html: str) -> dict[str, list[AnimePlanetVoiceActor]]:
    """Extract voice actors keyed by language code from a table-cell HTML block."""
    vas: dict[str, list[AnimePlanetVoiceActor]] = {}
    for m in _VA_FLAG_RE.finditer(cell_html):
        lang = _FLAG_LANG_MAP.get(m.group(1).upper(), m.group(1).lower())
        url = m.group(2)
        name = unescape(m.group(3).strip())
        if lang not in vas:
            vas[lang] = []
        vas[lang].append(AnimePlanetVoiceActor(name=name, url=url))
    return vas


def _extract_anime_roles(body_html: str) -> list[AnimePlanetCharacterAnimeRole]:
    """Extract anime ography entries from the 'Anime Roles' table."""
    section_match = _ANIME_ROLES_SECTION_RE.search(body_html)
    if not section_match:
        return []

    roles: list[AnimePlanetCharacterAnimeRole] = []
    for row_match in _TR_RE.finditer(section_match.group(1)):
        cells = _TD_RE.findall(row_match.group(1))
        if len(cells) < 2:
            continue
        title_cell = cells[0]
        role_cell = _strip_tags(cells[1])
        actors_cell = cells[2] if len(cells) > 2 else ""

        href_match = _OGRAPHY_HREF_RE.search(title_cell)
        title_match = _LAST_ANCHOR_TEXT_RE.search(title_cell)
        if not href_match or not title_match:
            continue

        roles.append(
            AnimePlanetCharacterAnimeRole(
                title=unescape(title_match.group(1).strip()),
                url=href_match.group(1),
                role=role_cell if role_cell else None,
                voice_actors=_extract_vas_from_cell(actors_cell),
            )
        )
    return roles


def _extract_manga_roles(body_html: str) -> list[AnimePlanetCharacterMangaRole]:
    """Extract manga ography entries from the 'Manga Roles' table."""
    section_match = _MANGA_ROLES_SECTION_RE.search(body_html)
    if not section_match:
        return []

    roles: list[AnimePlanetCharacterMangaRole] = []
    for row_match in _TR_RE.finditer(section_match.group(1)):
        cells = _TD_RE.findall(row_match.group(1))
        if len(cells) < 2:
            continue
        title_cell = cells[0]
        role_cell = _strip_tags(cells[1])

        href_match = _OGRAPHY_HREF_RE.search(title_cell)
        title_match = _LAST_ANCHOR_TEXT_RE.search(title_cell)
        if not href_match or not title_match:
            continue

        roles.append(
            AnimePlanetCharacterMangaRole(
                title=unescape(title_match.group(1).strip()),
                url=href_match.group(1),
                role=role_cell if role_cell else None,
            )
        )
    return roles


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_character_from_raw(raw: dict[str, Any], html: str, url: str) -> AnimePlanetCharacter:
    """Build AnimePlanetCharacter from XPath-extracted raw fields and full page HTML."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    bar = _extract_entry_bar(html)
    return AnimePlanetCharacter(
        name=(raw.get("name") or "").strip(),
        slug=slug,
        url=url,
        image=raw.get("image") or None,
        loved_rank=_parse_rank(raw.get("loved_rank")),
        hated_rank=_parse_rank(raw.get("hated_rank")),
        loved_count=_parse_loved_count(raw.get("loved_count")),
        gender=bar.get("gender"),
        hair_color=bar.get("hair_color"),
        description=_extract_description(html),
        tags=_extract_tags(html),
        alt_names=_extract_alt_names(html),
        attributes=_extract_metadata(html),
        anime_roles=_extract_anime_roles(html),
        manga_roles=_extract_manga_roles(html),
    )


# ---------------------------------------------------------------------------
# Cached single fetch
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_character_detail",
    dependencies=[_get_character_schema],
)
async def _fetch_character_data(url: str) -> dict[str, Any] | None:
    """Fetch a character detail page and extract raw fields. Cached by url.

    Returns dict with XPath-extracted fields plus ``_html`` key containing
    the full page HTML (used by all regex helpers). Cached by url.
    """
    results = await crawl_batch_urls(
        [url],
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(_get_character_schema()),
    )
    result = results[0] if results else None
    if not result:
        return None

    status = result.get("status_code")
    if status and status >= 400:
        logger.error(f"HTTP {status} for character {url}")
        return None
    if status and 300 <= status < 400:
        logger.debug(f"HTTP {status} (redirect followed) for character {url}")

    items: list[dict[str, Any]] = json.loads(result.get("extracted_content") or "[]")
    if not items:
        return None
    raw = items[0]
    raw["_html"] = result.get("html") or ""
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AnimePlanetCharacterCrawler(BaseCrawler[AnimePlanetCharacter, dict[str, Any]]):
    """Crawler for Anime-Planet character detail pages."""

    def get_extraction_schema(self) -> dict[str, Any]:
        return _get_character_schema()

    def normalize_identifier(self, identifier: str) -> str:
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_character_data(url)

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AnimePlanetCharacter:
        return _build_character_from_raw(processed_raw, processed_raw.get("_html") or "", url)

    def map_to_canonical(self, source_model: AnimePlanetCharacter) -> dict[str, Any]:
        return character_from_animeplanet(source_model)


async def fetch_animeplanet_character(url: str) -> dict[str, Any] | None:
    """Fetch a single Anime-Planet character detail page and return canonical dict.

    Args:
        url: Full character URL (e.g. https://www.anime-planet.com/characters/monkey-d-luffy).

    Returns:
        Canonical character dict on success, None on failure.
    """
    return await AnimePlanetCharacterCrawler(DockerTransport(), NullRepository()).crawl(
        url
    )


async def fetch_animeplanet_characters(
    urls: list[str],
    *,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any] | None]:
    """Fetch multiple character detail pages in a single batch Docker job.

    All URLs are submitted to Docker at once; processed at MAX_CONCURRENT_TASKS
    concurrency. Much faster than sequential single fetches for large casts.

    Args:
        urls: List of full character URLs (e.g. https://www.anime-planet.com/characters/luffy).
        on_result: Optional callback invoked with each successfully parsed
            canonical character dict as it completes (used for write-immediately streaming).

    Returns:
        List aligned to urls — None for any failed fetch.
    """
    if not urls:
        return []

    full_urls = urls
    logger.info(f"Batch fetching {len(full_urls)} AP character details...")

    cached_values, missing_indices = await _fetch_character_data.cache_batch_get(  # type: ignore[attr-defined]
        full_urls
    )

    characters: list[dict[str, Any] | None] = [None] * len(full_urls)

    for idx, cached in enumerate(cached_values):
        if cached is not None:
            html = cached.get("_html") or ""
            canonical = character_from_animeplanet(
                _build_character_from_raw(cached, html, full_urls[idx])
            )
            characters[idx] = canonical
            if on_result is not None:
                on_result(canonical)
        else:
            if idx not in missing_indices:
                missing_indices.append(idx)

    if not missing_indices:
        return characters

    missing_indices = sorted(set(missing_indices))
    missing_urls = [full_urls[i] for i in missing_indices]

    for offset in range(0, len(missing_urls), _CHARACTER_BATCH_SIZE):
        chunk_urls = missing_urls[offset : offset + _CHARACTER_BATCH_SIZE]
        chunk_indices = missing_indices[offset : offset + _CHARACTER_BATCH_SIZE]
        cache_values: list[dict[str, Any] | None] = [None] * len(chunk_urls)

        await get_ap_rate_limiter().acquire()
        results = await crawl_batch_urls(
            chunk_urls,
            browser_config=get_docker_browser_config(),
            crawler_config=get_docker_crawler_config(_get_character_schema()),
        )

        for idx_in_chunk, result in enumerate(results):
            out_index = chunk_indices[idx_in_chunk]
            if not result:
                characters[out_index] = None
                continue
            url = result.get("metadata", {}).get("og:url") or result["url"]
            status = result.get("status_code")
            if status and status >= 400:
                logger.error(f"HTTP {status} for character {url}")
                characters[out_index] = None
                continue
            if status and 300 <= status < 400:
                logger.debug(f"HTTP {status} (redirect followed) for character {url}")
            items: list[dict[str, Any]] = json.loads(
                result.get("extracted_content") or "[]"
            )
            if not items:
                characters[out_index] = None
                continue
            raw = items[0]
            page_html = result.get("html") or ""
            raw["_html"] = page_html  # store with raw for cache
            canonical = character_from_animeplanet(
                _build_character_from_raw(raw, page_html, url)
            )
            characters[out_index] = canonical
            cache_values[idx_in_chunk] = raw
            if on_result is not None:
                on_result(canonical)

        await _fetch_character_data.cache_batch_set(  # type: ignore[attr-defined]
            chunk_urls,
            cache_values,
        )

    return characters
