"""AniSearch Character Detail Crawler — zendriver + lxml XPath.

Two public functions:
    fetch_anisearch_character(url)   — single character detail page
    fetch_anisearch_characters(refs) — batch character detail pages

Character name, native name, image, description, and anime appearances are
extracted via lxml XPath on the raw page HTML. Voice actors (multi-language,
per-li language block) are extracted via regex on the full page HTML alongside
XPath fields.

A single persistent Chrome session is reused across all navigations in a batch
to avoid repeated browser startup overhead and to maintain session state.
"""

import asyncio
import logging
import re
from typing import Any, cast

from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchCharacter,
    AniSearchCharacterAnimeRole,
    AniSearchVoiceActorRef,
)
from enrichment.sources.anisearch.anisearch_mapper import character_from_anisearch
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
_INTER_REQUEST_DELAY = 3.0  # seconds between browser navigations
_CHARACTER_BATCH_SIZE = 20

# ---------------------------------------------------------------------------
# XPath selectors — direct lxml XPath, anchored on structural attributes
# ---------------------------------------------------------------------------

_XPATHS: dict[str, str] = {
    "name": "//h1[@id='htitle']",
    "name_native": (
        "//ul[contains(@class,'infoblock')]"
        "//div[@class='title'][@lang='ja']/span[@class='grey']"
    ),
    "image": "//img[@id='details-cover']/@src",
    "favorites": "//a[contains(@href,'/favorites')]//b",
    "tags": "//ul[contains(@class,'cloud')]//a[contains(@class,'gt')]",
    "description": (
        "//section[@id='description']"
        "//div[@lang='en'][contains(@class,'textblock')]"
    ),
    "screenshot_images": "//section[@id='images']//a[@class='loupe']/@href",
    "picture_images": "//section[@id='pictures']//img/@src",
    "anime_roles": "//section[@id='anime']//li//a[contains(@href,'anime/')]",
    # Ography sub-pages (/anime and /manga)
    "ography_entries": (
        "//ul[@class='covers']"
        "//a[contains(@href,'anime/') or contains(@href,'manga/')]"
    ),
}

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

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

_DESCRIPTION_PLACEHOLDER_RE = re.compile(
    r"would help many anime and manga fans", re.IGNORECASE
)
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")

# ---------------------------------------------------------------------------
# lxml extraction helpers
# ---------------------------------------------------------------------------


def _extract_character_from_html(html: str) -> dict[str, Any] | None:
    """Parse an AniSearch character page HTML into the raw field dict.

    Returns a dict with the same structure that crawl4ai schema extraction
    used to produce — same field names and value types — so all downstream
    helpers (_post_process_character, _build_character_from_raw, etc.) are
    unchanged. Returns None if the HTML cannot be parsed.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html.encode(), parser)
        if tree is None:
            return None
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    def _text(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        if not els:
            return None
        el = els[0]
        return " ".join(el.itertext()).strip() or None

    def _attr(key: str) -> str | None:
        vals = cast(list[str], tree.xpath(_XPATHS[key]))
        return vals[0].strip() if vals else None

    name = _text("name")
    name_native = _text("name_native")
    image = _attr("image")

    fav_els = cast(list[Any], tree.xpath(_XPATHS["favorites"]))
    favorites = " ".join(fav_els[0].itertext()).strip() if fav_els else None

    tag_els = cast(list[Any], tree.xpath(_XPATHS["tags"]))
    tags = [
        {"name": " ".join(el.itertext()).strip()}
        for el in tag_els
        if " ".join(el.itertext()).strip()
    ]

    desc_els = cast(list[Any], tree.xpath(_XPATHS["description"]))
    description = " ".join(desc_els[0].itertext()).strip() if desc_els else None

    screenshot_hrefs = cast(list[str], tree.xpath(_XPATHS["screenshot_images"]))
    screenshot_images = [{"url": h} for h in screenshot_hrefs if h]

    picture_srcs = cast(list[str], tree.xpath(_XPATHS["picture_images"]))
    picture_images = [{"url": s} for s in picture_srcs if s]

    role_els = cast(list[Any], tree.xpath(_XPATHS["anime_roles"]))
    anime_roles = []
    for el in role_els:
        href = el.get("href") or ""
        title_nodes = cast(list[Any], el.xpath(".//span[@class='title']"))
        title = " ".join(title_nodes[0].itertext()).strip() if title_nodes else ""
        if href or title:
            anime_roles.append({"url": href, "title": title})

    return {
        "name": name,
        "name_native": name_native,
        "image": image,
        "favorites": favorites,
        "tags": tags,
        "description": description,
        "screenshot_images": screenshot_images,
        "picture_images": picture_images,
        "anime_roles": anime_roles,
        "_html": html,
    }


def _extract_ography_from_html(html: str) -> list[dict[str, Any]] | None:
    """Parse an AniSearch /anime or /manga ography sub-page HTML.

    Returns a list of {url, title} dicts with absolute URLs, or None if the
    HTML cannot be parsed or contains no entries.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html.encode(), parser)
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    entry_els = cast(list[Any], tree.xpath(_XPATHS["ography_entries"]))
    entries = []
    for el in entry_els:
        href = el.get("href") or ""
        title_nodes = cast(list[Any], el.xpath(".//span[@class='title']"))
        title = " ".join(title_nodes[0].itertext()).strip() if title_nodes else ""
        if href and title:
            entries.append({
                "url": _absolutize_anime_url(href),
                "title": title,
            })
    return entries


# ---------------------------------------------------------------------------
# Browser navigation helper
# ---------------------------------------------------------------------------


async def _fetch_page_html(
    browser: Any, url: str, wait_selector: str | None = None
) -> str | None:
    """Navigate to url with an existing browser session and return page HTML.

    If wait_selector is given, waits for that CSS selector to appear in the DOM
    (up to 10s) instead of sleeping a fixed 2s. Falls back to a 2s sleep if no
    selector is provided.
    """
    try:
        page = await browser.get(url)
        if wait_selector:
            await page.wait_for(selector=wait_selector, timeout=10)
        else:
            await asyncio.sleep(2)
        return await page.get_content()
    except Exception as exc:
        logger.warning("navigation failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Regex helpers — voice actors and attributes
# ---------------------------------------------------------------------------


def _extract_voice_actors(html: str) -> list[AniSearchVoiceActorRef]:
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
    infoblock_match = _INFOBLOCK_RE.search(html)
    if not infoblock_match:
        return {}

    li_blocks: list[tuple[str, str]] = []
    for li_match in _INFOBLOCK_LI_RE.finditer(infoblock_match.group(1)):
        li_html = li_match.group(1)
        lang_match = _TITLE_LANG_RE.search(li_html)
        lang = lang_match.group(1).lower() if lang_match else ""
        li_blocks.append((lang, li_html))

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
    data = dict(raw)
    data["favorites"] = _parse_favorites(raw.get("favorites"))
    for role in data.get("anime_roles") or []:
        if role.get("url"):
            role["url"] = _absolutize_anime_url(role["url"])
    return data


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


def _build_character_from_raw(
    raw: dict[str, Any],
    html: str,
    url: str,
    role: str | None = None,
    anime_ography: list[dict[str, Any]] | None = None,
    manga_ography: list[dict[str, Any]] | None = None,
) -> AniSearchCharacter:
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
# Cached single-fetch functions (used by single-char path and cache layer)
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_character_detail",
    dependencies=[_extract_character_from_html],
)
async def _fetch_anisearch_character_data(url: str) -> dict[str, Any] | None:
    """Fetch a character detail page and extract raw fields. Cached by URL.

    Opens a temporary browser session — for high-volume use prefer the batch
    path in fetch_anisearch_characters which reuses a single session.
    """
    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        html = await _fetch_page_html(browser, url, wait_selector="#htitle")
        if not html:
            return None
        raw = _extract_character_from_html(html)
        if raw is None:
            return None
        return _post_process_character(raw)
    finally:
        try:
            await browser.stop()
        except Exception:  # noqa: S110
            pass


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_character_ography",
    dependencies=[_extract_ography_from_html],
)
async def _fetch_character_ography_data(url: str) -> list[dict[str, Any]] | None:
    """Fetch a single /anime or /manga ography sub-page. Cached by URL.

    Opens a temporary browser session — for high-volume use prefer the batch
    path which reuses a single session.
    """
    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        html = await _fetch_page_html(browser, url, wait_selector="#content")
        if not html:
            return None
        return _extract_ography_from_html(html)
    finally:
        try:
            await browser.stop()
        except Exception:  # noqa: S110
            pass


# ---------------------------------------------------------------------------
# Ography batch helper
# ---------------------------------------------------------------------------


async def _fetch_ography(
    url: str,
    browser: Any = None,
) -> list[dict[str, Any]] | None:
    """Fetch a single ography sub-page with cache check.

    Returns cached value if available. On a miss, navigates with browser if
    provided, otherwise opens a temporary session via _fetch_character_ography_data.
    """
    cached_values, missing_indices = await _fetch_character_ography_data.cache_batch_get(  # type: ignore[attr-defined]
        [url]
    )
    if not missing_indices:
        return cached_values[0]

    if browser is None:
        return await _fetch_character_ography_data(url)

    html = await _fetch_page_html(browser, url, wait_selector="#content")
    await asyncio.sleep(_INTER_REQUEST_DELAY)
    result = _extract_ography_from_html(html) if html else None
    await _fetch_character_ography_data.cache_batch_set([url], [result])  # type: ignore[attr-defined]
    return result


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

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

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
        return _build_character_from_raw(
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
    return await AniSearchCharacterCrawler(DockerTransport(), repo, role=role).crawl(url)


async def fetch_anisearch_characters(
    refs: list[dict[str, str]],
    *,
    output_path: str | None = None,
) -> list[dict[str, Any] | None]:
    """Batch-fetch character detail pages (+ ography sub-pages) for all refs.

    Uses a single persistent Chrome session for all browser navigations.
    Cache hits skip browser navigation entirely. Each character is written
    to output_path as soon as it is resolved.

    Args:
        refs: List of {"url": str, "role": str} dicts from fetch_anisearch_character_refs().
        output_path: If provided, each canonical character dict is appended as a
            JSONL line to this file as it completes.

    Returns:
        List aligned to refs — None for any failed fetch.
    """
    if not refs:
        return []

    urls = [r["url"] for r in refs]
    logger.info(f"Batch fetching {len(urls)} AniSearch character details...")
    repo = FileRepository(output_path) if output_path else NullRepository()
    characters: list[dict[str, Any] | None] = [None] * len(urls)

    # ── Batch cache lookup ────────────────────────────────────────────────
    cached_values, missing_indices = await _fetch_anisearch_character_data.cache_batch_get(  # type: ignore[attr-defined]
        urls
    )
    missing_set = set(missing_indices)

    import zendriver as zd

    browser: Any = None
    succeeded = 0

    try:
        for i, url in enumerate(urls):
            role = refs[i].get("role")

            # ── Detail page ───────────────────────────────────────────────
            if i not in missing_set:
                raw = cached_values[i]
            else:
                if browser is None:
                    browser = await zd.start(headless=False)
                html = await _fetch_page_html(browser, url, wait_selector="#htitle")
                if html is None:
                    await _fetch_anisearch_character_data.cache_batch_set(  # type: ignore[attr-defined]
                        [url], [None]
                    )
                    await asyncio.sleep(_INTER_REQUEST_DELAY)
                    continue
                extracted = _extract_character_from_html(html)
                raw = _post_process_character(extracted) if extracted else None
                await _fetch_anisearch_character_data.cache_batch_set(  # type: ignore[attr-defined]
                    [url], [raw]
                )
                await asyncio.sleep(_INTER_REQUEST_DELAY)

            if raw is None:
                continue

            # ── Ography sub-pages (sequential — shared browser, one tab) ────
            # If the detail page was a cache hit (browser=None), check ography cache
            # upfront so we can init one shared browser rather than letting
            # _batch_fetch_ography open a short-lived session per miss.
            if browser is None:
                _, anime_missing = await _fetch_character_ography_data.cache_batch_get(  # type: ignore[attr-defined]
                    [f"{url}/anime"]
                )
                _, manga_missing = await _fetch_character_ography_data.cache_batch_get(  # type: ignore[attr-defined]
                    [f"{url}/manga"]
                )
                if anime_missing or manga_missing:
                    browser = await zd.start(headless=False)
            anime_ography = await _fetch_ography(f"{url}/anime", browser)
            manga_ography = await _fetch_ography(f"{url}/manga", browser)

            # ── Build and save ────────────────────────────────────────────
            canonical = character_from_anisearch(
                _build_character_from_raw(
                    raw,
                    raw.get("_html") or "",
                    url,
                    role=role,
                    anime_ography=anime_ography,
                    manga_ography=manga_ography,
                )
            )
            characters[i] = canonical
            repo.save(canonical)
            succeeded += 1

    finally:
        if browser is not None:
            try:
                await browser.stop()
            except Exception:  # noqa: S110
                pass

    logger.info(
        "anisearch character fetch: %d/%d succeeded, %d cache hits",
        succeeded,
        len(urls),
        len(urls) - len(missing_set),
    )
    return characters
