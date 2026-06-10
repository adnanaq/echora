"""Anime-Planet Character Detail Crawler — zendriver + lxml XPath.

Two public functions:
    fetch_animeplanet_character(url)   — single character detail page
    fetch_animeplanet_characters(urls) — batch character detail pages

All data (name, description, tags, alt names, voice actors, ography) is
extracted from the character detail page via lxml XPath + Python regex helpers.
"""

import asyncio
import logging
import re
from html import unescape
from typing import Any, cast

from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
    AnimePlanetCharacterAnimeRole,
    AnimePlanetCharacterMangaRole,
    AnimePlanetVoiceActor,
)
from enrichment.sources.anime_planet.animeplanet_mapper import (
    character_from_animeplanet,
)
from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_URL = "https://www.anime-planet.com"

_CHARACTER_BATCH_SIZE = 20
_INTER_REQUEST_DELAY = 1.5

# XPaths for the five fields extracted via lxml (all others via regex on raw HTML)
_XPATHS: dict[str, str] = {
    "name": "//h1[@itemprop='name']",
    "image": "//img[@itemprop='image']/@src",
    # entryBar scope avoids matching the nav-menu anchors
    "loved_rank": "//section[contains(@class,'entryBar')]//a[contains(@href,'/characters/top-loved')]",
    "hated_rank": "//section[contains(@class,'entryBar')]//a[contains(@href,'/characters/top-hated')]",
    "loved_count": "//section[contains(@class,'sidebarStats')]//h3[contains(@class,'smSidebar')][.//span[@class='heartOn']]",
}

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
# Regex helper functions (unchanged)
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
# lxml extraction
# ---------------------------------------------------------------------------


def _extract_character_from_html(html: str) -> dict[str, Any] | None:
    """Extract raw character fields from a rendered Anime-Planet character page.

    Combines lxml XPath extraction (5 structured fields) with the full HTML
    stored under ``_html`` for use by all regex helpers.  The slug and URL are
    injected by the caller.

    Args:
        html: Full rendered HTML of an Anime-Planet character page.

    Returns:
        Raw dict with XPath fields and ``_html`` key, or None if the page has
        no character name.
    """
    if not html:
        return None

    tree = etree.fromstring(html, etree.HTMLParser(encoding="utf-8"))

    def _t(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        return "".join(els[0].itertext()).strip() if els else None

    def _a(key: str) -> str | None:
        vals = cast(list[Any], tree.xpath(_XPATHS[key]))
        return vals[0] if vals else None

    name = _t("name")
    if not name:
        return None

    return {
        "name": name,
        "image": _a("image"),
        "loved_rank": _t("loved_rank"),
        "hated_rank": _t("hated_rank"),
        "loved_count": _t("loved_count"),
        "_html": html,
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_character_from_raw(
    raw: dict[str, Any], html: str, url: str
) -> AnimePlanetCharacter:
    """Build AnimePlanetCharacter from extracted raw fields and full page HTML.

    Args:
        raw: Dict with XPath-extracted fields (name, image, loved_rank, etc.).
        html: Full page HTML for regex-based extraction.
        url: Canonical character URL (used to derive the slug).

    Returns:
        Validated AnimePlanetCharacter source model.
    """
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
# HTML fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_page_html(browser: Any, url: str) -> str | None:
    """Fetch a single character page using an existing zendriver browser session.

    Args:
        browser: Active zendriver browser instance.
        url: Full Anime-Planet character URL.

    Returns:
        Rendered page HTML, or None on navigation failure.
    """
    try:
        page = await browser.get(url)
        await page.wait_for(selector="h1[itemprop='name']", timeout=20)
        return await page.get_content()
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Cached single fetch
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_character_detail",
    dependencies=[_extract_character_from_html],
)
async def _fetch_character_data(url: str) -> dict[str, Any] | None:
    """Fetch a character detail page and extract raw fields. Cached by url.

    Returns dict with lxml-extracted fields plus ``_html`` key containing
    the full page HTML (used by all regex helpers).

    Args:
        url: Full Anime-Planet character URL.

    Returns:
        Raw extraction dict, or None on failure.
    """
    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        html = await _fetch_page_html(browser, url)
        if not html:
            logger.error(f"No HTML for character {url}")
            return None
        return _extract_character_from_html(html)
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AnimePlanetCharacterCrawler(BaseCrawler[AnimePlanetCharacter, dict[str, Any]]):
    """Crawler for Anime-Planet character detail pages."""

    def get_extraction_schema(self) -> dict[str, str]:
        return _XPATHS

    def normalize_identifier(self, identifier: str) -> str:
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_character_data(url)

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AnimePlanetCharacter:
        return _build_character_from_raw(
            processed_raw, processed_raw.get("_html") or "", url
        )

    def map_to_canonical(self, source_model: AnimePlanetCharacter) -> dict[str, Any]:
        return character_from_animeplanet(source_model)


async def fetch_animeplanet_character(url: str) -> dict[str, Any] | None:
    """Fetch a single Anime-Planet character detail page and return canonical dict.

    Args:
        url: Full character URL
            (e.g. ``https://www.anime-planet.com/characters/monkey-d-luffy``).

    Returns:
        Canonical character dict on success, None on failure.
    """
    return await AnimePlanetCharacterCrawler(NullRepository()).crawl(url)


async def fetch_animeplanet_characters(
    urls: list[str],
    *,
    output_path: str | None = None,
) -> list[dict[str, Any] | None]:
    """Fetch multiple character detail pages using a shared zendriver browser session.

    Cache hits are served immediately.  Cache misses are fetched sequentially
    via a single shared browser with a brief inter-request delay to avoid
    rate-limiting.

    Args:
        urls: List of full character URLs.
        output_path: If provided, each canonical character dict is appended as a
            JSONL line to this file as it completes.

    Returns:
        List aligned to ``urls`` — None for any failed fetch.
    """
    if not urls:
        return []

    repo = FileRepository(output_path) if output_path else NullRepository()
    logger.info(f"Batch fetching {len(urls)} AP character details...")

    cached_values, missing_indices = await _fetch_character_data.cache_batch_get(  # type: ignore[attr-defined]
        urls
    )

    characters: list[dict[str, Any] | None] = [None] * len(urls)

    for idx, cached in enumerate(cached_values):
        if cached is not None:
            html = cached.get("_html") or ""
            canonical = character_from_animeplanet(
                _build_character_from_raw(cached, html, urls[idx])
            )
            characters[idx] = canonical
            repo.save(canonical)
        else:
            if idx not in missing_indices:
                missing_indices.append(idx)

    if not missing_indices:
        return characters

    missing_indices = sorted(set(missing_indices))
    missing_urls = [urls[i] for i in missing_indices]
    cache_values: list[dict[str, Any] | None] = [None] * len(missing_urls)

    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        for i, url in enumerate(missing_urls):
            if i > 0:
                await asyncio.sleep(_INTER_REQUEST_DELAY)
            out_index = missing_indices[i]
            html = await _fetch_page_html(browser, url)
            if not html:
                characters[out_index] = None
                continue
            raw = _extract_character_from_html(html)
            if not raw:
                characters[out_index] = None
                continue
            canonical = character_from_animeplanet(
                _build_character_from_raw(raw, html, url)
            )
            characters[out_index] = canonical
            cache_values[i] = raw
            repo.save(canonical)
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")

    await _fetch_character_data.cache_batch_set(  # type: ignore[attr-defined]
        missing_urls,
        cache_values,
    )

    return characters
