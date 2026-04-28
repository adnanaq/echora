"""AniSearch Character Detail Crawler.

Two public functions:
    fetch_anisearch_character(url)   — single character detail page
    fetch_anisearch_characters(refs) — batch character detail pages

Character name, native name, image, description, and anime appearances are
extracted via XPath. Voice actors (multi-language, per-li language block)
are extracted via regex on the full page HTML, stored alongside XPath fields.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchCharacter,
    AniSearchCharacterAnimeRole,
    AniSearchVoiceActorRef,
)
from enrichment.sources.anisearch.anisearch_mapper import character_from_anisearch
from enrichment.sources.base.crawl4ai_docker import crawl_batch_urls
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
from enrichment.sources.base.framework import (
    BaseCrawler,
    DockerTransport,
    FileRepository,
    IRepository,
    ITransport,
    NullRepository,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

_ANISEARCH_BASE_URL = "https://www.anisearch.com"
_CHARACTER_BATCH_SIZE = 20

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

# Isolate the infoblock ul so we only iterate its li elements
_INFOBLOCK_RE = re.compile(
    r'<ul[^>]+class="[^"]*\binfoblock\b[^"]*"[^>]*>(.*?)</ul>',
    re.DOTALL | re.IGNORECASE,
)
_INFOBLOCK_LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.DOTALL | re.IGNORECASE)
_TITLE_LANG_RE = re.compile(r'<div[^>]+class="title"[^>]+lang="([^"]+)"', re.IGNORECASE)
_SEIYUU_LINK_RE = re.compile(
    r'<a[^>]+href="(person/[^"?#]+)"[^>]*>\s*([^<]+)\s*</a>',
    re.IGNORECASE,
)

# Maps ISO country code (from img alt / lang attr) to human-readable language name
_LANG_CODE_MAP: dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}

# Placeholder text AniSearch shows when no description is entered
_DESCRIPTION_PLACEHOLDER_RE = re.compile(
    r"would help many anime and manga fans", re.IGNORECASE
)
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# XPath schema
# ---------------------------------------------------------------------------


def _get_character_schema() -> dict[str, Any]:
    """XPath schema for AniSearch character detail page.

    Simple fields extracted via XPath; full HTML stored as _html for
    regex-based voice actor extraction across language blocks.
    """
    return {
        "name": "AniSearchCharacterDetail",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "name",
                "selector": "//h1[@id='htitle']",
                "type": "text",
            },
            {
                # Direct child span.grey of div.title[lang=ja] — the native name.
                # Sibling span.grey elements exist inside spoiler spans (deeper in the tree).
                "name": "name_native",
                "selector": "//ul[contains(@class,'infoblock')]//div[@class='title'][@lang='ja']/span[@class='grey']",
                "type": "text",
            },
            {
                "name": "image",
                "selector": "//img[@id='details-cover']",
                "type": "attribute",
                "attribute": "src",
            },
            {
                # Favourites count: <span class="afake">Favourites</span><b>677</b>
                "name": "favorites",
                "selector": "//a[contains(@href,'/favorites')]//b",
                "type": "text",
            },
            {
                "name": "tags",
                "selector": "//ul[contains(@class,'cloud')]//a[contains(@class,'gt')]",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "description",
                "selector": "//section[@id='description']//div[@lang='en'][contains(@class,'textblock')]",
                "type": "text",
            },
            {
                # Screenshots — full-size URLs from the loupe anchor hrefs.
                "name": "screenshot_images",
                "selector": "//section[@id='images']//a[@class='loupe']",
                "type": "list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    }
                ],
            },
            {
                # More presentations (manga covers, game art) — direct img src.
                "name": "picture_images",
                "selector": "//section[@id='pictures']//img",
                "type": "list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "src",
                    }
                ],
            },
            {
                # Anime appearances from the swiper section — title and relative URL.
                "name": "anime_roles",
                "selector": "//section[@id='anime']//li//a[contains(@href,'anime/')]",
                "type": "list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "title",
                        "selector": ".//span[@class='title']",
                        "type": "text",
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Ography sub-page schema
# ---------------------------------------------------------------------------


def _get_ography_schema() -> dict[str, Any]:
    """XPath schema for /anime and /manga character sub-pages.

    Both pages share ul.covers with per-item anchor + span.title.
    """
    return {
        "name": "AniSearchCharacterOgraphy",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "entries",
                "selector": "//ul[@class='covers']//a[contains(@href,'anime/') or contains(@href,'manga/')]",
                "type": "list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "title",
                        "selector": ".//span[@class='title']",
                        "type": "text",
                    },
                ],
            }
        ],
    }


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_character_ography",
    dependencies=[_get_ography_schema],
)
async def _fetch_character_ography_data(url: str) -> list[dict[str, Any]] | None:
    """Fetch a single /anime or /manga ography sub-page. Cached by URL."""
    results = await crawl_batch_urls(
        [url],
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(_get_ography_schema()),
    )
    result = results[0] if results else None
    if not result:
        return None
    status = result.get("status_code")
    if status and status >= 400:
        logger.error(f"HTTP {status} for ography {url}")
        return None
    items: list[dict[str, Any]] = json.loads(result.get("extracted_content") or "[]")
    if not items:
        return None
    return [
        {
            "url": _absolutize_anime_url(e["url"]),
            "title": (e.get("title") or "").strip(),
        }
        for e in (items[0].get("entries") or [])
        if e.get("url") and (e.get("title") or "").strip()
    ]


def _parse_ography_result(
    result: dict[str, Any] | None, sub_url: str
) -> list[dict[str, Any]] | None:
    """Parse a raw crawl result for an ography sub-page."""
    if not result:
        return None
    status = result.get("status_code")
    if status and status >= 400:
        logger.error(f"HTTP {status} for ography {sub_url}")
        return None
    items: list[dict[str, Any]] = json.loads(result.get("extracted_content") or "[]")
    if not items:
        return None
    return [
        {
            "url": _absolutize_anime_url(e["url"]),
            "title": (e.get("title") or "").strip(),
        }
        for e in (items[0].get("entries") or [])
        if e.get("url") and (e.get("title") or "").strip()
    ]


def _ography_to_roles(
    entries: list[dict[str, Any]] | None,
) -> list[AniSearchCharacterAnimeRole]:
    if not entries:
        return []
    return [
        AniSearchCharacterAnimeRole(title=e["title"], url=e["url"])
        for e in entries
        if e.get("title")
    ]


# ---------------------------------------------------------------------------
# Regex helper — voice actors
# ---------------------------------------------------------------------------


def _extract_voice_actors(html: str) -> list[AniSearchVoiceActorRef]:
    """Parse voice actor entries from the infoblock HTML.

    Each li in the infoblock represents one language. Language is read from the
    div.title[lang] attribute; VA links come from the adjacent div.seiyuu.
    """
    infoblock_match = _INFOBLOCK_RE.search(html)
    if not infoblock_match:
        return []

    vas: list[AniSearchVoiceActorRef] = []
    for li_match in _INFOBLOCK_LI_RE.finditer(infoblock_match.group(1)):
        li_html = li_match.group(1)
        lang_match = _TITLE_LANG_RE.search(li_html)
        if not lang_match:
            continue
        lang_code = lang_match.group(1).lower()
        language = _LANG_CODE_MAP.get(lang_code, lang_code.capitalize())

        for link_match in _SEIYUU_LINK_RE.finditer(li_html):
            href = link_match.group(1).strip()
            name = link_match.group(2).strip()
            if not name or not href:
                continue
            url = f"{_ANISEARCH_BASE_URL}/{href.lstrip('/')}"
            vas.append(AniSearchVoiceActorRef(name=name, language=language, url=url))

    return vas


_ATTR_EXCLUDED_CLASSES = frozenset({"title", "seiyuu", "anime", "manga"})


def _extract_attributes(html: str) -> dict[str, str]:
    """Extract character attribute divs from all infoblock li blocks.

    English li values take priority; missing keys are filled from other lis.
    Key = div class (spaces → '_'). Value = inner text with tags stripped, label removed.
    """
    infoblock_match = _INFOBLOCK_RE.search(html)
    if not infoblock_match:
        return {}

    li_blocks: list[tuple[str, str]] = []  # (lang_code, li_html)
    for li_match in _INFOBLOCK_LI_RE.finditer(infoblock_match.group(1)):
        li_html = li_match.group(1)
        lang_match = _TITLE_LANG_RE.search(li_html)
        lang = lang_match.group(1).lower() if lang_match else ""
        li_blocks.append((lang, li_html))

    # English li first, then the rest
    li_blocks.sort(key=lambda t: (0 if t[0] == "en" else 1))

    def _attrs_from_li(li_html: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for div_match in re.finditer(
            r'<div\s+class="([^"]+)"[^>]*>(.*?)</div>',
            li_html,
            re.DOTALL | re.IGNORECASE,
        ):
            css_class = div_match.group(1).strip()
            if css_class in _ATTR_EXCLUDED_CLASSES:
                continue
            key = css_class.replace(" ", "_")
            raw_text = _STRIP_TAGS_RE.sub("", div_match.group(2)).strip()
            value = raw_text.split(":", 1)[-1].strip()
            if key and value:
                result[key] = value
        return result

    attrs: dict[str, str] = {}
    for _, li_html in li_blocks:
        for key, value in _attrs_from_li(li_html).items():
            if key not in attrs:
                attrs[key] = value
    return attrs


# ---------------------------------------------------------------------------
# Post-processing and model builder
# ---------------------------------------------------------------------------


def _parse_favorites(raw: str | None) -> int | None:
    if not raw:
        return None
    m = re.search(r"[\d,]+", raw)
    return int(m.group().replace(",", "")) if m else None


def _absolutize_anime_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"{_ANISEARCH_BASE_URL}/{href.lstrip('/')}"


def _post_process_character(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse favorites, absolutize URLs. Mutates a copy of raw."""
    data = dict(raw)
    data["favorites"] = _parse_favorites(raw.get("favorites"))
    for role in data.get("anime_roles") or []:
        if role.get("url"):
            role["url"] = _absolutize_anime_url(role["url"])
    return data


def _build_character(
    raw: dict[str, Any],
    html: str,
    url: str,
    role: str | None = None,
    anime_ography: list[dict[str, Any]] | None = None,
    manga_ography: list[dict[str, Any]] | None = None,
) -> AniSearchCharacter:
    """Construct AniSearchCharacter from XPath-extracted fields and full page HTML."""
    description = (raw.get("description") or "").strip() or None
    if description and _DESCRIPTION_PLACEHOLDER_RE.search(description):
        description = None

    anime_roles = [
        AniSearchCharacterAnimeRole(
            title=(r.get("title") or "").strip(),
            url=r.get("url") or None,
        )
        for r in raw.get("anime_roles") or []
        if (r.get("title") or "").strip()
    ]

    tags = [
        t["name"].strip()
        for t in (raw.get("tags") or [])
        if (t.get("name") or "").strip()
    ]
    screenshot_images = [
        i["url"] for i in (raw.get("screenshot_images") or []) if i.get("url")
    ]
    picture_images = [
        i["url"] for i in (raw.get("picture_images") or []) if i.get("url")
    ]

    return AniSearchCharacter(
        source=url,
        name=(raw.get("name") or "").strip() or None,
        name_native=(raw.get("name_native") or "").strip() or None,
        image=raw.get("image") or None,
        favorites=raw.get("favorites"),
        description=description,
        role=role,
        tags=tags,
        screenshot_images=screenshot_images,
        picture_images=picture_images,
        voice_actors=_extract_voice_actors(html),
        anime_roles=anime_roles,
        anime_ography=_ography_to_roles(anime_ography),
        manga_ography=_ography_to_roles(manga_ography),
        attributes=_extract_attributes(html),
    )


# ---------------------------------------------------------------------------
# Cached single fetch
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_character_detail",
    dependencies=[_get_character_schema],
)
async def _fetch_anisearch_character_data(url: str) -> dict[str, Any] | None:
    """Fetch a character detail page and extract raw fields. Cached by URL.

    Returns a dict of XPath-extracted fields plus ``_html`` (full page HTML
    for regex-based voice actor extraction).
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
        logger.debug(f"HTTP {status} (redirect) for character {url}")

    items: list[dict[str, Any]] = json.loads(result.get("extracted_content") or "[]")
    if not items:
        return None

    raw = _post_process_character(items[0])
    raw["_html"] = result.get("html") or ""
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AniSearchCharacterCrawler(BaseCrawler[AniSearchCharacter, dict[str, Any]]):
    """Crawler for AniSearch character detail pages."""

    def __init__(
        self,
        transport: ITransport,
        repository: IRepository | None = None,
        *,
        role: str | None = None,
    ) -> None:
        super().__init__(transport, repository)
        self._role = role

    def normalize_identifier(self, identifier: str) -> str:
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_anisearch_character_data(url)

    async def post_process_raw_data(
        self, raw_data: dict[str, Any], url: str
    ) -> dict[str, Any]:
        anime_ography, manga_ography = await asyncio.gather(
            _fetch_character_ography_data(f"{url}/anime"),
            _fetch_character_ography_data(f"{url}/manga"),
        )
        return {
            **raw_data,
            "_anime_ography": anime_ography,
            "_manga_ography": manga_ography,
        }

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AniSearchCharacter:
        return _build_character(
            processed_raw,
            processed_raw.get("_html") or "",
            url,
            role=self._role,
            anime_ography=processed_raw.get("_anime_ography"),
            manga_ography=processed_raw.get("_manga_ography"),
        )

    def map_to_canonical(self, source_model: AniSearchCharacter) -> dict[str, Any]:
        return character_from_anisearch(source_model)


async def fetch_anisearch_character(
    url: str,
    *,
    role: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a single AniSearch character detail page and return canonical dict.

    Args:
        url: Full character URL (e.g. https://www.anisearch.com/character/4852,monkey-d-luffy).
        role: Role string from the refs list ("Main Character", "Secondary Character", etc.)
        output_path: If provided, append the canonical dict as a JSONL line to this path.

    Returns:
        Canonical character dict on success, None on failure.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await AniSearchCharacterCrawler(DockerTransport(), repo, role=role).crawl(
        url
    )


async def _batch_fetch_ography(
    sub_urls: list[str],
) -> list[list[dict[str, Any]] | None]:
    """Batch-fetch ography sub-pages with cache_batch_get/set.

    sub_urls: list of absolute /anime or /manga URLs.
    Returns list aligned to sub_urls.
    """
    (
        cached_values,
        missing_indices,
    ) = await _fetch_character_ography_data.cache_batch_get(  # type: ignore[attr-defined]
        sub_urls
    )
    results: list[list[dict[str, Any]] | None] = list(cached_values)

    if missing_indices:
        missing_urls = [sub_urls[i] for i in missing_indices]
        raw_results = await crawl_batch_urls(
            missing_urls,
            browser_config=get_docker_browser_config(),
            crawler_config=get_docker_crawler_config(_get_ography_schema()),
        )
        cache_values: list[list[dict[str, Any]] | None] = []
        for i, result in enumerate(raw_results):
            parsed = _parse_ography_result(result, missing_urls[i])
            results[missing_indices[i]] = parsed
            cache_values.append(parsed)
        await _fetch_character_ography_data.cache_batch_set(  # type: ignore[attr-defined]
            missing_urls, cache_values
        )

    return results


async def fetch_anisearch_characters(
    refs: list[dict[str, str]],
    *,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any] | None]:
    """Batch-fetch character detail pages (+ ography sub-pages) for all refs.

    Args:
        refs: List of {"url": str, "role": str} dicts from fetch_anisearch_character_refs().
        on_result: Optional callback invoked with each canonical character dict
            as it is parsed (write-immediately streaming).

    Returns:
        List aligned to refs — None for any failed fetch.
    """
    if not refs:
        return []

    urls = [r["url"] for r in refs]
    logger.info(f"Batch fetching {len(urls)} AniSearch character details...")

    # ── Step 1: detail pages ──────────────────────────────────────────────
    (
        cached_values,
        missing_indices,
    ) = await _fetch_anisearch_character_data.cache_batch_get(  # type: ignore[attr-defined]
        urls
    )

    raw_data: list[dict[str, Any] | None] = list(cached_values)

    missing_indices = sorted(set(missing_indices))
    missing_urls = [urls[i] for i in missing_indices]

    for offset in range(0, len(missing_urls), _CHARACTER_BATCH_SIZE):
        chunk_urls = missing_urls[offset : offset + _CHARACTER_BATCH_SIZE]
        chunk_indices = missing_indices[offset : offset + _CHARACTER_BATCH_SIZE]
        cache_values_detail: list[dict[str, Any] | None] = [None] * len(chunk_urls)

        results = await crawl_batch_urls(
            chunk_urls,
            browser_config=get_docker_browser_config(),
            crawler_config=get_docker_crawler_config(_get_character_schema()),
        )

        for idx_in_chunk, result in enumerate(results):
            out_index = chunk_indices[idx_in_chunk]
            if not result:
                continue
            url = result.get("url") or chunk_urls[idx_in_chunk]
            status = result.get("status_code")
            if status and status >= 400:
                logger.error(f"HTTP {status} for character {url}")
                continue
            if status and 300 <= status < 400:
                logger.debug(f"HTTP {status} (redirect) for character {url}")
            items: list[dict[str, Any]] = json.loads(
                result.get("extracted_content") or "[]"
            )
            if not items:
                continue
            raw = _post_process_character(items[0])
            raw["_html"] = result.get("html") or ""
            raw_data[out_index] = raw
            cache_values_detail[idx_in_chunk] = raw

        await _fetch_anisearch_character_data.cache_batch_set(  # type: ignore[attr-defined]
            chunk_urls, cache_values_detail
        )

    # ── Step 2: ography sub-pages (all characters in one batch) ──────────
    anime_sub_urls = [f"{url}/anime" for url in urls]
    manga_sub_urls = [f"{url}/manga" for url in urls]
    anime_ography_list, manga_ography_list = (
        await _batch_fetch_ography(anime_sub_urls),
        await _batch_fetch_ography(manga_sub_urls),
    )

    # ── Step 3: build canonical characters ───────────────────────────────
    characters: list[dict[str, Any] | None] = [None] * len(urls)
    for idx, raw in enumerate(raw_data):
        if raw is None:
            continue
        url = urls[idx]
        role = refs[idx].get("role")
        canonical = character_from_anisearch(
            _build_character(
                raw,
                raw.get("_html") or "",
                url,
                role=role,
                anime_ography=anime_ography_list[idx],
                manga_ography=manga_ography_list[idx],
            )
        )
        characters[idx] = canonical
        if on_result is not None:
            on_result(canonical)

    return characters
