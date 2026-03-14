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
from pathlib import Path
from typing import Any

from enrichment.crawlers.crawl4ai_docker import crawl_batch_urls, crawl_single_url
from enrichment.crawlers.mal_crawler.mal_base import (
    MAL_BASE_URL,
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
from enrichment.crawlers.utils import sanitize_output_path
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = get_mal_scraping_limiter()


def _get_character_schema() -> dict[str, Any]:
    """CSS schema for MAL character detail pages."""
    return {
        "name": "MalCharacterDetail",
        "baseSelector": "body",
        "fields": [
            {
                "name": "name_header",
                "selector": "h2.normal_header",
                "type": "text",
            },
            {
                "name": "title",
                "selector": "title",
                "type": "text",
            },
            {
                "name": "image_src",
                "selector": "td.borderClass img",
                "type": "attribute",
                "attribute": "data-src",
            },
            {
                "name": "image_src_fallback",
                "selector": "td.borderClass img",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "content_html",
                "selector": "div#content",
                "type": "html",
            },
        ],
    }


def _extract_name_and_native(
    name_header: str | None, title: str | None
) -> tuple[str, str | None]:
    """Extract canonical name and native (kanji) name from MAL character page.

    MAL format in h2: "Monkey D., Luffy (モンキー・D・ルフィ)"
    Or in title: "Monkey D., Luffy | MyAnimeList.net"
    """
    raw = name_header or title or ""
    raw = re.sub(r"\s*\|\s*MyAnimeList\.net.*$", "", raw).strip()

    native_match = re.search(r"\(([^\)]+)\)\s*$", raw)
    native = native_match.group(1).strip() if native_match else None
    name = raw[: native_match.start()].strip() if native_match else raw.strip()

    return name or raw, native


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

    bio_html = bio_section.group(1)

    # Two-pass spoiler strip: inner content first, then outer wrapper
    bio_html = re.sub(
        r'<(?:div|span)[^>]*class="[^"]*spoiler_content[^"]*"[^>]*>.*?</(?:div|span)>',
        "",
        bio_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    bio_html = re.sub(
        r'<(?:div|span)[^>]*class="[^"]*spoiler[^"]*"[^>]*>.*?</(?:div|span)>',
        "",
        bio_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    bio_html = re.sub(r"<input[^>]*>", "", bio_html, flags=re.IGNORECASE)

    for line in re.split(r"<br\s*/?>", bio_html, flags=re.IGNORECASE):
        text = re.sub(r"<[^>]+>", "", line).strip()
        if ":" in text:
            key, _, value = text.partition(":")
            key = key.strip()
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
        for link_match in re.finditer(
            r'<a[^>]*href="[^"]*myanimelist[^"]*/people/(\d+)/[^"]*"[^>]*>(.*?)</a>',
            row_html, re.DOTALL,
        ):
            candidate = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
            if candidate:
                person_id = int(link_match.group(1))
                name = candidate
                break
        if not person_id or not name:
            continue

        lang_match = re.search(r"<small[^>]*>(.*?)</small>", row_html, re.DOTALL | re.IGNORECASE)
        language = re.sub(r"<[^>]+>", "", lang_match.group(1)).strip() if lang_match else ""

        img_match = re.search(r'<img[^>]*(?:data-src|src)="([^"]+)"', row_html)
        image_url = img_match.group(1) if img_match else None

        results.append(MalVoiceActorRef(person_id=person_id, name=name, language=language, image_url=image_url))

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
async def _fetch_mal_character_data(char_id: int) -> dict[str, Any] | None:
    """Fetch /character/{id} and extract character detail. Cached by char_id."""
    url = f"{MAL_BASE_URL}/character/{char_id}"

    await _limiter.acquire()
    result = await crawl_single_url(
        url=url,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(
            _get_character_schema(),
            strategy_type="JsonCssExtractionStrategy",
            # magic=True,  # TEST: disabled to check if this causes ~23s slowdown
            # simulate_user=False,  # TEST: disabled to check simulation overhead
        ),
    )
    if not result:
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for character {char_id}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    if not raw_list:
        return None
    raw = raw_list[0]
    raw["_char_id"] = char_id
    raw["_url"] = url
    return raw


def _parse_character_raw(raw: dict[str, Any], char_id: int) -> MalScrapedCharacter:
    """Parse a raw extraction dict into a MalScrapedCharacter."""
    saved_id = raw.pop("_char_id", char_id)
    saved_url = raw.pop("_url", f"{MAL_BASE_URL}/character/{char_id}")

    name, name_native = _extract_name_and_native(
        raw.get("name_header"), raw.get("title")
    )

    image_url = raw.get("image_src") or raw.get("image_src_fallback") or ""
    images = [image_url] if image_url else []

    content_html = raw.get("content_html") or ""

    fav_match = re.search(r"Member\s+Favorites:\s*([\d,]+)", content_html, re.IGNORECASE)
    favorites = parse_number(fav_match.group(1)) if fav_match else 0

    nicknames_raw = parse_sidebar_field(content_html, "Nicknames")
    nicknames = [n.strip() for n in nicknames_raw.split(",")] if nicknames_raw else []

    return MalScrapedCharacter(
        mal_id=saved_id,
        url=saved_url,
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


async def fetch_mal_character(
    char_id: int,
    output_path: str | None = None,
) -> MalScrapedCharacter | None:
    """Fetch a single MAL character detail page.

    Args:
        char_id: MAL character ID (e.g., 40 for Luffy).
        output_path: Optional path to write JSON output.

    Returns:
        MalScrapedCharacter if successful, None otherwise.
    """
    logger.info(f"[character] Fetching character {char_id}...")
    raw = await _fetch_mal_character_data(char_id)
    if not raw:
        logger.error(f"[character] Failed to fetch character {char_id}")
        return None

    char = _parse_character_raw(raw, char_id)

    if output_path:
        from enrichment.mappers.mal_mapper import character_from_mal
        safe_path = sanitize_output_path(output_path)
        canonical = character_from_mal(char)
        Path(safe_path).write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Character {char_id} saved to {safe_path}")

    return char


async def fetch_mal_characters(
    char_ids: list[int],
) -> list[MalScrapedCharacter | None]:
    """Fetch multiple character detail pages in a single batch Docker job.

    All URLs are submitted at once; Docker processes them at MAX_CONCURRENT_TASKS
    concurrency. Much faster than sequential single fetches for large casts.

    Args:
        char_ids: List of MAL character IDs.

    Returns:
        List aligned to char_ids — None for any failed fetch.
    """
    if not char_ids:
        return []

    logger.info(f"[characters] Batch fetching {len(char_ids)} character details...")
    urls = [f"{MAL_BASE_URL}/character/{cid}" for cid in char_ids]

    results = await crawl_batch_urls(
        urls,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(
            _get_character_schema(),
            strategy_type="JsonCssExtractionStrategy",
            # magic=True,  # TEST: disabled to check if this causes ~23s slowdown
            # simulate_user=False,  # TEST: disabled to check simulation overhead
        ),
    )

    characters: list[MalScrapedCharacter | None] = []
    for char_id, result in zip(char_ids, results):
        if not result:
            characters.append(None)
            continue
        status = result.get("status_code")
        if status and status != 200:
            logger.error(f"[character] HTTP {status} for character {char_id}")
            characters.append(None)
            continue
        raw_list = json.loads(result.get("extracted_content") or "[]")
        if not raw_list:
            characters.append(None)
            continue
        raw = raw_list[0]
        raw["_char_id"] = char_id
        raw["_url"] = f"{MAL_BASE_URL}/character/{char_id}"
        characters.append(_parse_character_raw(raw, char_id))

    return characters


async def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Fetch MAL character data")
    subparsers = parser.add_subparsers(dest="command")

    ids_parser = subparsers.add_parser("ids", help="Fetch character refs from anime page")
    ids_parser.add_argument("mal_id", type=int, help="MAL anime ID")
    ids_parser.add_argument("anime_url", type=str, help="Canonical anime URL")

    detail_parser = subparsers.add_parser("detail", help="Fetch character detail page")
    detail_parser.add_argument("char_id", type=int, help="MAL character ID")
    detail_parser.add_argument("--output", type=str, default="mal_character.json")

    args = parser.parse_args()

    if args.command == "ids":
        from enrichment.crawlers.mal_crawler.mal_character_refs_crawler import fetch_mal_character_refs
        refs = await fetch_mal_character_refs(args.mal_id, args.anime_url)
        logger.info("Found %s characters", len(refs))
        for ref in refs[:5]:
            logger.info("  %s (ID=%s, role=%s)", ref.name, ref.char_id, ref.role)
    elif args.command == "detail":
        char = await fetch_mal_character(args.char_id, output_path=args.output)
        if char is None:
            logger.error("No data for character %s", args.char_id)
            return 1
        logger.info("Done: %s", char.name)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
