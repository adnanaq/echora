"""MAL character detail crawler — zendriver + lxml XPath.

CLI usage::

    uv run python -m enrichment.sources.mal.mal_character_crawler <url> [--output path]

Example::

    uv run python -m enrichment.sources.mal.mal_character_crawler \\
        https://myanimelist.net/character/40/Luffy_Monkey_D --output luffy.json
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import Any, cast

from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from enrichment.sources.mal.mal_base import (
    parse_number,
    parse_sidebar_field,
)
from enrichment.sources.mal.mal_mapper import character_from_mal
from enrichment.sources.mal.mal_models import (
    MalCharacter,
    MalOgraphyEntry,
    MalVoiceActorRef,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_INTER_REQUEST_DELAY = 3.0

# ---------------------------------------------------------------------------
# XPath selectors — anchored on structural attributes rather than CSS class
# names, which change frequently on MAL.
# ---------------------------------------------------------------------------

_XPATHS: dict[str, str] = {
    # Character name — h2.normal_header: "Name (NativeName)" on all MAL
    # character pages; only element providing both names in one extraction.
    "name_header": "//h2[contains(@class,'normal_header')]",
    # Character portrait image in the fixed-width left sidebar
    "image_src": (
        "//td[@width='225' and contains(@class,'borderClass')]"
        "//img[contains(@class,'portrait')]/@data-src"
    ),
    # Favorites count td in the left column
    "favorites_td": "//td[contains(normalize-space(),'Member Favorites:')]",
    # Full content block — parsed by regex helpers for bio, ography, VA sections
    "content": "//div[@id='content']",
}


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def _extract_character_from_html(html: str) -> dict[str, Any] | None:
    """Parse a MAL character detail page into the raw field dict.

    Args:
        html: Full HTML of a MAL character detail page.

    Returns:
        Dict with ``name_header``, ``image_src``, ``favorites``, and
        ``content_html`` keys, or None if the page cannot be parsed.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.fromstring(html.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    name_els = cast(list[Any], tree.xpath(_XPATHS["name_header"]))
    name_header = "".join(name_els[0].itertext()).strip() if name_els else None

    img_vals = cast(list[str], tree.xpath(_XPATHS["image_src"]))
    image_src = img_vals[0].strip() if img_vals else None

    fav_tds = cast(list[Any], tree.xpath(_XPATHS["favorites_td"]))
    favorites: str | None = None
    if fav_tds:
        fav_text = "".join(fav_tds[0].itertext())
        m = re.search(r"Member Favorites:\s*([\d,]+)", fav_text)
        favorites = m.group(1) if m else None

    content_els = cast(list[Any], tree.xpath(_XPATHS["content"]))
    content_html = (
        etree.tostring(content_els[0], encoding="unicode", method="html")
        if content_els
        else None
    )

    if not name_header and not content_html:
        return None

    return {
        "name_header": name_header,
        "image_src": image_src,
        "favorites": favorites,
        "content_html": content_html or "",
    }


# ---------------------------------------------------------------------------
# Post-processing helpers (pure transforms — operate on raw content_html)
# ---------------------------------------------------------------------------


def _extract_name_and_native(
    name_header: str | None,
) -> tuple[str, str | None]:
    """Extract canonical name and native (kanji) name from MAL character page.

    MAL format in h2.normal_header: "Monkey D., Luffy (モンキー・D・ルフィ)"
    """
    raw = (name_header or "").strip()

    native_match = re.search(r"\(([^\)]+)\)\s*$", raw)
    native = native_match.group(1).strip() if native_match else None
    name = raw[: native_match.start()].strip() if native_match else raw.strip()

    return name or raw, native


def _extract_spoiler(div_html: str) -> str:
    """Extract the revealed text from a single <div class="spoiler"> block."""
    sc = re.search(
        r"<span[^>]*spoiler_content[^>]*>.*?<br\s*/?>(.*?)</span>",
        div_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not sc:
        return ""
    return re.sub(r"<[^>]+>", "", sc.group(1)).strip()


def _bio_section_html(content_html: str) -> str | None:
    """Return the raw HTML of the first normal_header bio section."""
    m = re.search(
        r"<h2[^>]*class=\"[^\"]*normal_header[^\"]*\"[^>]*>.*?</h2>(.*?)"
        r"(?:<h2|<div[^>]*class=\"[^\"]*normal_header|$)",
        content_html,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else None


def _tokenize_spoilers(html: str, spoiler_map: dict[str, str]) -> str:
    """Replace each spoiler div with a unique placeholder token.

    Populates spoiler_map {token: revealed_text} and returns the processed HTML
    with all spoiler divs replaced. The internal <br> inside spoiler_content is
    eliminated, making subsequent <br>-based line splitting safe.
    """

    def _replace(m: re.Match[str]) -> str:
        token = f"__SPOILER_{len(spoiler_map)}__"
        spoiler_map[token] = _extract_spoiler(m.group(0))
        return token

    processed = re.sub(
        r'<div[^>]*class="[^"]*spoiler[^"]*"[^>]*>.*?</div>',
        _replace,
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return re.sub(r"<input[^>]*>", "", processed, flags=re.IGNORECASE)


def _extract_bio_data(content_html: str) -> tuple[dict[str, str], dict[str, str]]:
    """Extract key:value biographical data pairs, split into (attributes, spoilers).

    attributes — non-spoiler values only
    spoilers   — spoiler values keyed by the same field names
    """
    attributes: dict[str, str] = {}
    spoilers: dict[str, str] = {}

    raw_html = _bio_section_html(content_html)
    if not raw_html:
        return attributes, spoilers

    spoiler_map: dict[str, str] = {}
    processed = _tokenize_spoilers(raw_html, spoiler_map)

    for line in re.split(r"<br\s*/?>", processed, flags=re.IGNORECASE):
        tokens = re.findall(r"__SPOILER_\d+__", line)
        clean = re.sub(r"__SPOILER_\d+__", "", line)
        clean = re.sub(r"<[^>]+>", "", clean).strip().rstrip(",").strip()

        if ":" not in clean:
            continue
        key, _, value = clean.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not key or len(key) >= 50:
            continue

        if value:
            attributes[key] = value
        for token in tokens:
            spoiler_text = spoiler_map.get(token, "")
            if spoiler_text:
                spoilers[key] = spoiler_text

    return attributes, spoilers


def _extract_description(content_html: str) -> tuple[str | None, str | None]:
    """Extract description text, split into (description, description_spoiler).

    description         — non-spoiler prose (same content as before)
    description_spoiler — text hidden inside prose-level spoiler divs; None if absent

    Bio-field spoiler lines (short key before colon) are excluded from
    description_spoiler — they are already captured by _extract_bio_data.
    """
    raw_html = _bio_section_html(content_html)
    if not raw_html:
        return None, None

    spoiler_map: dict[str, str] = {}
    processed = _tokenize_spoilers(raw_html, spoiler_map)

    prose_spoiler_parts: list[str] = []
    for line in re.split(r"<br\s*/?>", processed, flags=re.IGNORECASE):
        tokens = re.findall(r"__SPOILER_\d+__", line)
        if not tokens:
            continue
        clean = re.sub(r"__SP_\d+__", "", line)
        clean = re.sub(r"<[^>]+>", "", clean).strip()
        colon_pos = clean.find(":")
        if 0 < colon_pos < 30:
            continue  # bio-field line — skip
        for token in tokens:
            text = spoiler_map.get(token, "")
            if text:
                prose_spoiler_parts.append(text)

    description_spoiler = " ".join(" ".join(prose_spoiler_parts).split()) or None

    # --- description: non-spoiler prose (strip all spoiler divs) ---
    section_html = raw_html
    section_html = re.sub(
        r'<(?:div|span)[^>]*class="[^"]*spoiler_content[^"]*"[^>]*>.*?</(?:div|span)>',
        "",
        section_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    section_html = re.sub(
        r'<(?:div|span)[^>]*class="[^"]*spoiler[^"]*"[^>]*>.*?</(?:div|span)>',
        "",
        section_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    section_html = re.sub(r"<input[^>]*>", "", section_html, flags=re.IGNORECASE)

    desc_lines = []
    for line in re.split(r"<br\s*/?>", section_html, flags=re.IGNORECASE):
        text = re.sub(r"<[^>]+>", "", line).strip()
        if not text:
            continue
        colon_pos = text.find(":")
        if 0 < colon_pos < 30:
            continue
        desc_lines.append(text)

    description = " ".join(" ".join(desc_lines).split()) or None
    return description, description_spoiler


def _extract_voice_actors(content_html: str) -> list[MalVoiceActorRef]:
    """Extract voice actors from the 'Voice Actors' section of a character detail page."""
    va_section_match = re.search(
        r'<div[^>]*class="[^"]*normal_header[^"]*"[^>]*>\s*Voice Actors\s*</div>(.*?)$',
        content_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not va_section_match:
        return []

    section_html = va_section_match.group(1)
    results: list[MalVoiceActorRef] = []

    for tr_match in re.finditer(
        r"<tr[^>]*>(.*?)</tr>", section_html, re.DOTALL | re.IGNORECASE
    ):
        row_html = tr_match.group(1)

        person_id: int | None = None
        name = ""
        source_url = ""
        for link_match in re.finditer(
            r'<a[^>]*href="([^"]*myanimelist[^"]*/people/(\d+)/[^"]*)"[^>]*>(.*?)</a>',
            row_html,
            re.DOTALL,
        ):
            candidate = re.sub(r"<[^>]+>", "", link_match.group(3)).strip()
            if candidate:
                source_url = link_match.group(1)
                person_id = int(link_match.group(2))
                name = candidate
                break
        if not person_id or not name:
            continue

        lang_match = re.search(
            r"<small[^>]*>(.*?)</small>", row_html, re.DOTALL | re.IGNORECASE
        )
        language = (
            re.sub(r"<[^>]+>", "", lang_match.group(1)).strip() if lang_match else ""
        )

        img_match = re.search(r'<img[^>]*(?:data-src|src)="([^"]+)"', row_html)
        image_url = img_match.group(1) if img_match else None

        results.append(
            MalVoiceActorRef(
                person_id=person_id,
                name=name,
                language=language,
                image_url=image_url,
                sources=[source_url] if source_url else [],
            )
        )

    return results


def _extract_ography(content_html: str, section: str) -> list[MalOgraphyEntry]:
    """Extract anime or manga ography entries from character content HTML."""
    results: list[MalOgraphyEntry] = []
    section_match = re.search(
        rf'<div[^>]*class="[^"]*normal_header[^"]*"[^>]*>\s*{re.escape(section)}\s*</div>(.*?)'
        r'(?:<div[^>]*class="[^"]*normal_header|$)',
        content_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return results

    for row_match in re.finditer(
        r"<tr[^>]*>(.*?)</tr>", section_match.group(1), re.DOTALL | re.IGNORECASE
    ):
        row_html = row_match.group(1)

        url = title = ""
        for link_match in re.finditer(
            r'<a[^>]*href="([^"]*(?:anime|manga)/\d+[^"]*)"[^>]*>(.*?)</a>',
            row_html,
            re.DOTALL,
        ):
            title = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
            if title:
                url = link_match.group(1)
                break
        if not title:
            continue

        role: str | None = None
        role_match = re.search(
            r"<small[^>]*>(.*?)</small>", row_html, re.DOTALL | re.IGNORECASE
        )
        if role_match:
            role_text = re.sub(r"<[^>]+>", "", role_match.group(1)).strip()
            if role_text:
                role = role_text

        results.append(MalOgraphyEntry(title=title, role=role, sources=[url]))

    return results


# ---------------------------------------------------------------------------
# Browser navigation helper
# ---------------------------------------------------------------------------


async def _fetch_character_html(browser: Any, url: str) -> tuple[str, str] | None:
    """Navigate to a MAL character URL and return (html, canonical_url).

    The Voice Actors section uses intersection-observer lazy loading — it only
    renders when scrolled into view. scroll_down triggers it before capture.

    Args:
        browser: Active zendriver browser instance.
        url: MAL character URL.

    Returns:
        Tuple of (rendered HTML, canonical URL after redirect), or None on failure.
    """
    try:
        page = await browser.get(url)
        await page.wait_for(selector="h2.normal_header", timeout=10)
        await page.scroll_down(amount=1000, speed=3000)
        await asyncio.sleep(2)
        return await page.get_content(), page.url or url
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_character_detail",
    dependencies=[_extract_character_from_html],
)
async def _fetch_mal_character_data(url: str) -> tuple[dict[str, Any], str] | None:
    """Fetch a MAL character detail page via zendriver. Cached by URL.

    Args:
        url: Full MAL character URL.

    Returns:
        Tuple of (raw extraction dict, canonical URL), or None on failure.
    """
    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        result = await _fetch_character_html(browser, url)
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")

    if result is None:
        return None

    html, canonical_url = result
    raw = _extract_character_from_html(html)
    if raw is None:
        logger.warning(f"extraction failed for character {url}")
        return None

    return raw, canonical_url


def _build_character_from_raw(raw: dict[str, Any], url: str) -> MalCharacter:
    """Parse a raw extraction dict into a MalCharacter."""
    name, name_native = _extract_name_and_native(raw.get("name_header"))

    image_url = raw.get("image_src") or ""
    images = [image_url] if image_url else []

    content_html = raw.get("content_html") or ""

    favorites = parse_number(raw.get("favorites") or "") or 0

    nicknames_raw = parse_sidebar_field(content_html, "Nicknames")
    nicknames = [n.strip() for n in nicknames_raw.split(",")] if nicknames_raw else []

    attrs, spoilers = _extract_bio_data(content_html)
    description, description_spoiler = _extract_description(content_html)
    if description_spoiler:
        spoilers["description"] = description_spoiler

    return MalCharacter(
        source=url,
        name=name,
        name_native=name_native,
        description=description,
        nicknames=nicknames,
        favorites=favorites or 0,
        images=images,
        character_info=attrs,
        spoilers=spoilers,
        animeography=_extract_ography(content_html, "Animeography"),
        mangaography=_extract_ography(content_html, "Mangaography"),
        voice_actors=_extract_voice_actors(content_html),
    )


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class MalCharacterCrawler(BaseCrawler[MalCharacter, dict[str, Any]]):
    """Crawler for MyAnimeList character detail pages.

    Uses zendriver for browser automation and lxml XPath for extraction.
    """

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        result = await _fetch_mal_character_data(url)
        if result is None:
            return None
        raw, canonical_url = result
        return {"_raw": raw, "_canonical_url": canonical_url}

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> MalCharacter:
        return _build_character_from_raw(
            processed_raw["_raw"], processed_raw["_canonical_url"]
        )

    def map_to_canonical(self, source_model: MalCharacter) -> dict[str, Any]:
        return character_from_mal(source_model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_mal_character(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch a single MAL character detail page and return canonical dict.

    Args:
        url: Full MAL character URL (e.g. https://myanimelist.net/character/40/Luffy).
        output_path: If provided, append the canonical dict as a JSONL line.

    Returns:
        Canonical character dict, or None on failure.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await MalCharacterCrawler(repo).crawl(url)


async def fetch_mal_characters(
    urls: list[str],
    *,
    output_path: str | None = None,
) -> list[dict[str, Any] | None]:
    """Fetch multiple character detail pages in a single shared browser session.

    Cache is checked upfront for all URLs in one round-trip. Hits are returned
    immediately. Misses are fetched sequentially in one persistent zendriver
    session with inter-request delays. Each result is cached immediately so
    progress is not lost on cancellation.

    Args:
        urls: List of full MAL character URLs.
        output_path: If provided, each canonical character dict is appended as a
            JSONL line to this file as it completes.

    Returns:
        List aligned to urls — None for any failed fetch.
    """
    if not urls:
        return []

    logger.info(f"Batch fetching {len(urls)} MAL character details...")
    repo = FileRepository(output_path) if output_path else NullRepository()

    cached_values, missing_indices = await _fetch_mal_character_data.cache_batch_get(  # type: ignore[attr-defined]
        urls
    )

    characters: list[dict[str, Any] | None] = [None] * len(urls)

    def _parse_cached(value: Any) -> dict[str, Any] | None:
        if not value:
            return None
        if isinstance(value, list | tuple) and len(value) == 2:
            raw, canonical_url = value
        else:
            return None
        if not isinstance(raw, dict) or not canonical_url:
            return None
        return character_from_mal(_build_character_from_raw(raw, canonical_url))

    for idx, cached in enumerate(cached_values):
        parsed = _parse_cached(cached)
        if parsed is not None:
            characters[idx] = parsed
            repo.save(parsed)
        else:
            if idx not in missing_indices:
                missing_indices.append(idx)

    if not missing_indices:
        return characters

    missing_indices = sorted(set(missing_indices))

    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        for i, idx in enumerate(missing_indices):
            url = urls[idx]
            result = await _fetch_character_html(browser, url)
            if result is None:
                characters[idx] = None
                continue

            html, canonical_url = result
            raw = _extract_character_from_html(html)
            if raw is None:
                logger.warning(f"extraction failed for character {url}")
                characters[idx] = None
                continue

            # Cache immediately so cancellation doesn't lose progress
            await _fetch_mal_character_data.cache_batch_set(  # type: ignore[attr-defined]
                [url], [(raw, canonical_url)]
            )

            canonical = character_from_mal(
                _build_character_from_raw(raw, canonical_url)
            )
            characters[idx] = canonical
            repo.save(canonical)

            if i < len(missing_indices) - 1:
                await asyncio.sleep(_INTER_REQUEST_DELAY)
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")

    return characters


async def main() -> int:
    """CLI entry point for fetching a MAL character page.

    Returns:
        0 on success, 1 if extraction fails.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Fetch MAL character data")
    parser.add_argument(
        "url",
        type=str,
        help="MAL character URL (e.g. https://myanimelist.net/character/40/Luffy)",
    )
    parser.add_argument(
        "--output", type=str, default="mal_character.json", help="Output file path"
    )
    args = parser.parse_args()

    char = await fetch_mal_character(args.url, output_path=args.output)
    if char is None:
        logger.error(f"No data for character {args.url}")
        return 1
    logger.info(f"Done: {char.get('name')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
