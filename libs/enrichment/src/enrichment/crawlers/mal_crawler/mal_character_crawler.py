"""MAL Character Detail Crawler.

Two public functions:
    fetch_mal_character(char_id)   — /character/{id}       → MalScrapedCharacter | None
    fetch_mal_characters(char_ids) — batch /character/{id} → list[MalScrapedCharacter | None]

Usage:
    from enrichment.crawlers.mal_crawler.mal_character_crawler import (
        fetch_mal_character,
        fetch_mal_characters,
    )
    char = await fetch_mal_character(40)               # Luffy character detail
    chars = await fetch_mal_characters([40, 41, 42])   # batch
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Any

from enrichment.crawlers.crawl4ai_docker import crawl_batch_urls
from enrichment.crawlers.mal_crawler.mal_base import (
    get_mal_docker_browser_config,
    get_mal_docker_crawler_config,
    get_mal_scraping_limiter,
    parse_number,
    parse_sidebar_field,
)
from enrichment.crawlers.mal_crawler.mal_models import (
    MalOgraphyEntry,
    MalScrapedCharacter,
    MalVoiceActorRef,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = get_mal_scraping_limiter()


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


def _inline_spoilers(html: str) -> str:
    """Replace spoiler divs with their revealed text content.

    MAL spoiler structure:
        <div class="spoiler ...">
          <input ... value="Show">
          <span class="spoiler_content" style="display:none">
            <input ... value="Hide"><br>
            spoiler text here
          </span>
        </div>

    The visible text follows the <br> inside spoiler_content. We extract that
    text and substitute the whole spoiler div with it, preserving field values.
    """

    def _replace(m: re.Match[str]) -> str:
        sc = re.search(
            r'<span[^>]*spoiler_content[^>]*>.*?<br\s*/?>(.*?)</span>',
            m.group(0),
            re.DOTALL | re.IGNORECASE,
        )
        if not sc:
            return ""
        return re.sub(r"<[^>]+>", "", sc.group(1)).strip()

    result = re.sub(
        r'<div[^>]*class="[^"]*spoiler[^"]*"[^>]*>.*?</div>',
        _replace,
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    result = re.sub(r"<input[^>]*>", "", result, flags=re.IGNORECASE)
    result = re.sub(r",\s*,", ",", result)
    result = re.sub(r":\s*,", ":", result)
    return result


def _extract_bio_data(content_html: str) -> dict[str, str]:
    """Extract key:value biographical data pairs from MAL character content."""
    bio: dict[str, str] = {}

    bio_section = re.search(
        r"<h2[^>]*class=\"[^\"]*normal_header[^\"]*\"[^>]*>.*?</h2>(.*?)"
        r"(?:<h2|<div[^>]*class=\"[^\"]*normal_header|$)",
        content_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not bio_section:
        return bio

    bio_html = _inline_spoilers(bio_section.group(1))

    for line in re.split(r"<br\s*/?>", bio_html, flags=re.IGNORECASE):
        text = re.sub(r"<[^>]+>", "", line).strip()
        if ":" in text:
            key, _, value = text.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key and value and len(key) < 50:
                bio[key] = value

    return bio


def _extract_description(content_html: str) -> str | None:
    """Extract the description text from character content HTML."""
    bio_section = re.search(
        r'<h2[^>]*class="[^"]*normal_header[^"]*"[^>]*>.*?</h2>(.*?)'
        r'(?:<div[^>]*class="[^"]*normal_header|<h2|$)',
        content_html,
        re.DOTALL | re.IGNORECASE,
    )
    if not bio_section:
        return None

    section_html = bio_section.group(1)

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

    description = " ".join(" ".join(desc_lines).split())
    return description if description else None


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

    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", section_html, re.DOTALL | re.IGNORECASE):
        row_html = tr_match.group(1)

        person_id: int | None = None
        name = ""
        source_url = ""
        for link_match in re.finditer(
            r'<a[^>]*href="([^"]*myanimelist[^"]*/people/(\d+)/[^"]*)"[^>]*>(.*?)</a>',
            row_html, re.DOTALL,
        ):
            candidate = re.sub(r"<[^>]+>", "", link_match.group(3)).strip()
            if candidate:
                source_url = link_match.group(1)
                person_id = int(link_match.group(2))
                name = candidate
                break
        if not person_id or not name:
            continue

        lang_match = re.search(r"<small[^>]*>(.*?)</small>", row_html, re.DOTALL | re.IGNORECASE)
        language = re.sub(r"<[^>]+>", "", lang_match.group(1)).strip() if lang_match else ""

        img_match = re.search(r'<img[^>]*(?:data-src|src)="([^"]+)"', row_html)
        image_url = img_match.group(1) if img_match else None

        results.append(MalVoiceActorRef(
            person_id=person_id,
            name=name,
            language=language,
            image_url=image_url,
            sources=[source_url] if source_url else [],
        ))

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

    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", section_match.group(1), re.DOTALL | re.IGNORECASE):
        row_html = row_match.group(1)

        url = title = ""
        entry_id = 0
        for link_match in re.finditer(
            r'<a[^>]*href="([^"]*(?:anime|manga)/(\d+)[^"]*)"[^>]*>(.*?)</a>',
            row_html, re.DOTALL,
        ):
            title = re.sub(r"<[^>]+>", "", link_match.group(3)).strip()
            if title:
                url = link_match.group(1)
                entry_id = int(link_match.group(2))
                break
        if not title:
            continue

        role: str | None = None
        role_match = re.search(r"<small[^>]*>(.*?)</small>", row_html, re.DOTALL | re.IGNORECASE)
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
async def _fetch_mal_character_data(url: str) -> dict[str, Any] | None:
    """Fetch /character/{id} and extract character detail. Cached by url."""
    await _limiter.acquire()
    results = await crawl_batch_urls(
        [url],
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(_get_character_schema()),
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
    return raw_list[0]


def _parse_character_raw(raw: dict[str, Any], url: str) -> MalScrapedCharacter:
    """Parse a raw extraction dict into a MalScrapedCharacter."""
    name, name_native = _extract_name_and_native(raw.get("name_header"))

    image_url = raw.get("image_src") or ""
    images = [image_url] if image_url else []

    content_html = raw.get("content_html") or ""

    favorites = parse_number(raw.get("favorites") or "") or 0

    nicknames_raw = parse_sidebar_field(content_html, "Nicknames")
    nicknames = [n.strip() for n in nicknames_raw.split(",")] if nicknames_raw else []

    return MalScrapedCharacter(
        source=url,
        name=name,
        name_native=name_native,
        description=_extract_description(content_html),
        nicknames=nicknames,
        favorites=favorites or 0,
        images=images,
        character_info=_extract_bio_data(content_html),
        animeography=_extract_ography(content_html, "Animeography"),
        mangaography=_extract_ography(content_html, "Mangaography"),
        voice_actors=_extract_voice_actors(content_html),
    )


async def fetch_mal_character(url: str) -> MalScrapedCharacter | None:
    """Fetch a single MAL character detail page.

    Args:
        url: Full MAL character URL (e.g. https://myanimelist.net/character/40/Luffy).

    Returns:
        MalScrapedCharacter if successful, None otherwise.
    """
    logger.info(f"[character] Fetching character {url}...")
    raw = await _fetch_mal_character_data(url)
    if not raw:
        logger.error(f"[character] Failed to fetch character {url}")
        return None

    return _parse_character_raw(raw, url)


async def fetch_mal_characters(
    urls: list[str],
) -> list[MalScrapedCharacter | None]:
    """Fetch multiple character detail pages in a single batch Docker job.

    All URLs are submitted at once; Docker processes them at MAX_CONCURRENT_TASKS
    concurrency. Much faster than sequential single fetches for large casts.

    Args:
        urls: List of full MAL character URLs.

    Returns:
        List aligned to urls — None for any failed fetch.
    """
    if not urls:
        return []

    logger.info(f"[characters] Batch fetching {len(urls)} character details...")

    results = await crawl_batch_urls(
        urls,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(_get_character_schema()),
    )

    characters: list[MalScrapedCharacter | None] = []
    for result in results:
        if not result:
            characters.append(None)
            continue
        url = result["url"]
        status = result.get("status_code")
        if status and status != 200:
            logger.error(f"[character] HTTP {status} for character {url}")
            characters.append(None)
            continue
        raw_list = json.loads(result.get("extracted_content") or "[]")
        if not raw_list:
            characters.append(None)
            continue
        characters.append(_parse_character_raw(raw_list[0], url))

    return characters


async def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Fetch MAL character data")
    parser.add_argument("url", type=str, help="MAL character URL (e.g. https://myanimelist.net/character/40/Luffy)")
    parser.add_argument("--output", type=str, default="mal_character.json", help="Output file path")
    args = parser.parse_args()

    char = await fetch_mal_character(args.url)
    if char is None:
        logger.error(f"No data for character {args.url}")
        return 1
    from pathlib import Path

    from enrichment.mappers.mal_mapper import character_from_mal
    canonical = character_from_mal(char)
    Path(args.output).write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done: {char.name}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
