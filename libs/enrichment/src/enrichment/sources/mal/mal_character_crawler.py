"""MAL Character Detail Crawler.

Two public functions:
    fetch_mal_character(url)   — /character/{id}       → dict[str, Any] | None
    fetch_mal_characters(urls) — batch /character/{id} → list[dict[str, Any] | None]

Usage:
    from enrichment.sources.mal.mal_character_crawler import (
        fetch_mal_character,
        fetch_mal_characters,
    )
    char = await fetch_mal_character("https://myanimelist.net/character/40/Luffy")
    chars = await fetch_mal_characters([url1, url2, url3])
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable
from typing import Any

from enrichment.sources.base.crawl4ai_docker import crawl_batch_urls
from enrichment.sources.base.crawler_config import (
    CrawlerRateLimiter,
    get_docker_browser_config,
    get_docker_crawler_config,
)
from enrichment.sources.base.framework import (
    DockerTransport,
    FileRepository,
    NullRepository,
)
from enrichment.sources.mal.mal_base import (
    parse_number,
    parse_sidebar_field,
)
from enrichment.sources.mal.mal_base_crawler import MalCrawlerBase
from enrichment.sources.mal.mal_mapper import character_from_mal
from enrichment.sources.mal.mal_models import (
    MalOgraphyEntry,
    MalCharacter,
    MalVoiceActorRef,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = CrawlerRateLimiter(min_interval_seconds=10.0, max_per_minute=25)

_CHARACTER_BATCH_SIZE = 30


def _get_character_schema() -> dict[str, Any]:
    """XPath extraction schema for MAL character detail pages.

    Anchors on structural attributes (id, width) and text content rather than
    CSS class names, which change frequently. The bulk of parsing (bio, ography,
    voice actors) is done by Python helpers operating on the raw content_html block.
    """
    return {
        "name": "MalCharacterDetail",
        "baseSelector": "//body",
        "fields": [
            # Character name — h2.normal_header contains "Name (NativeName)" on all
            # MAL character pages. It is the only element that provides both the
            # canonical name and the native (kanji) name in a single extraction.
            # <title> and og:title live in <head>, unreachable from baseSelector //body.
            {
                "name": "name_header",
                "selector": "//h2[contains(@class,'normal_header')]",
                "type": "text",
            },
            # Character image — portrait class in the fixed-width left sidebar td
            # (no itemprop on character pages unlike anime pages)
            {
                "name": "image_src",
                "selector": "//td[@width='225' and contains(@class,'borderClass')]//img[contains(@class,'portrait')]",
                "type": "attribute",
                "attribute": "data-src",
            },
            # Favorites count — plain text node in the left column td
            {
                "name": "favorites",
                "selector": "//td[contains(normalize-space(),'Member Favorites:')]",
                "type": "regex",
                "pattern": r"Member Favorites:\s*([\d,]+)",
            },
            # Full content block — parsed by Python helpers for bio, ography, VA sections
            {
                "name": "content_html",
                "selector": "//div[@id='content']",
                "type": "html",
            },
        ],
    }


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


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_character_detail",
    dependencies=[
        _get_character_schema,
        _extract_name_and_native,
        _extract_bio_data,
        _extract_description,
        _extract_voice_actors,
        _extract_ography,
    ],
)
async def _fetch_mal_character_data(url: str) -> tuple[dict[str, Any], str] | None:
    """Fetch /character/{id} and extract character detail. Cached by url.

    Returns:
        (raw, canonical_url) on success, None on failure.
    """
    await _limiter.acquire()
    results = await crawl_batch_urls(
        [url],
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(_get_character_schema()),
    )
    result = results[0] if results else None
    if not result:
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for character {url}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    if not raw_list:
        return None
    canonical_url = result.get("metadata", {}).get("og:url") or url
    return raw_list[0], canonical_url


def _parse_character_raw(raw: dict[str, Any], url: str) -> MalCharacter:
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


class MalCharacterCrawler(MalCrawlerBase[MalCharacter, dict[str, Any]]):
    """Crawler for MyAnimeList character detail pages."""

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
        return _parse_character_raw(
            processed_raw["_raw"], processed_raw["_canonical_url"]
        )

    def map_to_canonical(self, source_model: MalCharacter) -> dict[str, Any]:
        return character_from_mal(source_model)


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
    return await MalCharacterCrawler(DockerTransport(), repo).crawl(url)


async def fetch_mal_characters(
    urls: list[str],
    *,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any] | None]:
    """Fetch multiple character detail pages in a single batch Docker job.

    All URLs are submitted at once; Docker processes them at MAX_CONCURRENT_TASKS
    concurrency. Much faster than sequential single fetches for large casts.

    Args:
        urls: List of full MAL character URLs.
        on_result: Optional callback invoked with each successfully parsed
            canonical character dict as results arrive (used for write-immediately streaming).

    Returns:
        List aligned to urls — None for any failed fetch.
    """
    if not urls:
        return []

    logger.info(f"Batch fetching {len(urls)} MAL character details...")

    cached_values, missing_indices = await _fetch_mal_character_data.cache_batch_get(  # type: ignore[attr-defined]
        urls
    )

    characters: list[dict[str, Any] | None] = [None] * len(urls)

    def _parse_cached(value: Any) -> dict[str, Any] | None:
        if not value:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            raw, canonical_url = value
        else:
            return None
        if not isinstance(raw, dict) or not canonical_url:
            return None
        return character_from_mal(_parse_character_raw(raw, canonical_url))

    for idx, cached in enumerate(cached_values):
        parsed = _parse_cached(cached)
        if parsed is not None:
            characters[idx] = parsed
            if on_result is not None:
                on_result(parsed)
        else:
            if idx not in missing_indices:
                missing_indices.append(idx)

    if not missing_indices:
        return characters

    missing_indices = sorted(set(missing_indices))
    missing_urls = [urls[i] for i in missing_indices]

    for offset in range(0, len(missing_urls), _CHARACTER_BATCH_SIZE):
        chunk_urls = missing_urls[offset : offset + _CHARACTER_BATCH_SIZE]
        chunk_indices = missing_indices[offset : offset + _CHARACTER_BATCH_SIZE]
        cache_values: list[tuple[dict[str, Any], str] | None] = [None] * len(chunk_urls)

        await _limiter.acquire()
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
            if status and status != 200:
                logger.error(f"HTTP {status} for character {url}")
                characters[out_index] = None
                continue
            raw_list = json.loads(result.get("extracted_content") or "[]")
            if not raw_list:
                characters[out_index] = None
                continue
            raw_for_cache = raw_list[0]
            canonical = character_from_mal(_parse_character_raw(raw_for_cache, url))
            characters[out_index] = canonical
            cache_values[idx_in_chunk] = (raw_for_cache, url)
            if on_result is not None:
                on_result(canonical)

        await _fetch_mal_character_data.cache_batch_set(  # type: ignore[attr-defined]
            chunk_urls,
            cache_values,
        )

    return characters


async def main() -> int:
    """Fetch a single MAL character and write the mapped result to JSON."""
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
