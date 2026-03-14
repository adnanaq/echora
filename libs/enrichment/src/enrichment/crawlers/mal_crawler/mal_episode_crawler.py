"""MAL Episode Detail Crawler.

Fetches /anime/{id}/episode/{num} from MyAnimeList and extracts episode metadata
into a MalScrapedEpisode model.

Episodes are community-contributed on MAL. Character/staff data is populated for
well-documented episodes but empty for many episodes of long-running shows (e.g.,
One Piece past ~ep 700). Empty sections return empty lists — no failure.

Usage:
    from enrichment.crawlers.mal_crawler.mal_episode_crawler import fetch_mal_episode

    ep = await fetch_mal_episode("anime/21/One_Piece/episode/1")
    ep = await fetch_mal_episode("https://myanimelist.net/anime/21/One_Piece/episode/1", output_path="ep1.json")
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from enrichment.crawlers.mal_crawler.mal_base import (
    MAL_BASE_URL,
    get_mal_docker_browser_config,
    get_mal_docker_crawler_config,
    get_mal_scraping_limiter,
    parse_duration_seconds,
    parse_iso_date,
)
from enrichment.crawlers.mal_crawler.mal_models import (
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalScrapedEpisode,
)
from enrichment.crawlers.utils import sanitize_output_path
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = get_mal_scraping_limiter()


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a single JSON record as a JSONL line. Logs and continues on failure."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    except Exception as e:
        logger.warning("Failed to append episode record to %s: %s", path, e)


def _get_episode_schema() -> dict[str, Any]:
    """XPath extraction schema for MAL episode detail pages.

    Key MAL episode page structure (verified March 2026 against One Piece ep 1):
    - Episode title: h2.fs18 (NOT div#content h2 — 6 sidebar h2s appear first)
    - Subtitle (romaji + kanji): p.fn-grey2
    - Aired/Duration: div.di-tc.ar (right-aligned info box in the episode header row)
    - Synopsis: text inside the Synopsis block (h2.fw-b parent div)
    - Characters/Staff: parsed from the section list container (table.fl-l entries)
    """
    return {
        "name": "MalEpisodeDetail",
        "baseSelector": "//body",
        "fields": [
            # Episode title h2 — class "fs18" is unique to the episode title h2.
            # The page has 6 other h2s (Alternative Titles, Information, Statistics,
            # Available At, Resources, Streaming Platforms) before the episode title.
            {
                "name": "title",
                "selector": "//h2[contains(@class,'fs18')]",
                "type": "regex",
                "pattern": r"#\d+\s*-\s*(.*?)(?:\s+\b(?:Filler|Recap)\b)?\s*$",
            },
            {
                "name": "is_filler",
                "selector": "//h2[contains(@class,'fs18')]",
                "type": "regex",
                "pattern": r"\b(Filler)\b",
            },
            {
                "name": "is_recap",
                "selector": "//h2[contains(@class,'fs18')]",
                "type": "regex",
                "pattern": r"\b(Recap)\b",
            },
            {
                "name": "title_japanese",
                "selector": "//p[contains(@class,'fn-grey2')]",
                "type": "regex",
                "pattern": r"\(([^)]+)\)\s*$",
            },
            {
                "name": "title_romaji",
                "selector": "//p[contains(@class,'fn-grey2')]",
                "type": "regex",
                "pattern": r"^(.*?)(?:\s*\([^)]+\))?\s*$",
            },
            {
                "name": "duration_raw",
                "selector": "//div[contains(@class,'di-tc') and contains(@class,'ar')]",
                "type": "regex",
                "pattern": r"Duration:\s*(\d{1,2}:\d{2}:\d{2})",
            },
            {
                "name": "aired_raw",
                "selector": "//div[contains(@class,'di-tc') and contains(@class,'ar')]",
                "type": "regex",
                "pattern": r"Aired:\s*([A-Za-z]+ \d{1,2},\s*\d{4})",
            },
            {
                "name": "synopsis_raw",
                "selector": "//h2[normalize-space()='Synopsis']/parent::div",
                "type": "regex",
                "pattern": r"Synopsis\s*(.*)",
            },
            {
                "name": "characters",
                "selector": (
                    "//table[contains(@class,'fl-l')]"
                    "[preceding::h2[1][contains(normalize-space(),'Characters')]]"
                ),
                "type": "nested_list",
                "fields": [
                    {
                        "name": "char_name",
                        "selector": ".//a[contains(@class,'fw-b')][contains(@href,'/character/')]",
                        "type": "text",
                    },
                    {
                        "name": "char_url",
                        "selector": ".//a[contains(@class,'fw-b')][contains(@href,'/character/')]",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "role",
                        "selector": ".",
                        "type": "regex",
                        "pattern": r"\b(Main|Supporting)\b",
                    },
                    {
                        "name": "voice_actors_html",
                        "selector": ".//p[contains(@class,'pb8')]",
                        "type": "html",
                    },
                ],
            },
            {
                "name": "staff",
                "selector": (
                    "//table[contains(@class,'fl-l')]"
                    "[preceding::h2[1][normalize-space()='Staff']]"
                ),
                "type": "nested_list",
                "fields": [
                    {
                        "name": "name",
                        "selector": ".//a[contains(@class,'fw-b')][contains(@href,'/people/')]",
                        "type": "text",
                    },
                    {
                        "name": "person_url",
                        "selector": ".//a[contains(@class,'fw-b')][contains(@href,'/people/')]",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "role",
                        "selector": ".//p[contains(@class,'pr12')]",
                        "type": "text",
                    },
                ],
            },
        ],
    }


def _parse_title_info(
    header_text: str | None,
    subtitle_text: str | None,
    episode_number: int,
) -> tuple[str, str | None, str | None, bool, bool]:
    """Parse MAL episode header into (title, title_japanese, title_romaji, is_filler, is_recap).

    Args:
        header_text: Raw text of the h2.fs18 element, e.g. "#1 - I'm Luffy! Filler"
        subtitle_text: Raw text of p.fn-grey2, e.g. "Romaji Title (Japanese Title)"
        episode_number: Fallback episode number for title when header_text is None.

    Returns:
        Tuple of (title, title_japanese, title_romaji, is_filler, is_recap).
    """
    is_filler = False
    is_recap = False
    title = f"Episode {episode_number}"

    if header_text:
        # Strip "#N - " prefix
        m = re.match(r"^#\d+\s*-\s*(.*)", header_text)
        raw_title = m.group(1) if m else header_text

        # Detect and strip filler/recap suffix (case-insensitive)
        filler_m = re.search(r"\s+\bFiller\b\s*$", raw_title, re.IGNORECASE)
        recap_m = re.search(r"\s+\bRecap\b\s*$", raw_title, re.IGNORECASE)
        if filler_m:
            is_filler = True
            raw_title = raw_title[: filler_m.start()]
        elif recap_m:
            is_recap = True
            raw_title = raw_title[: recap_m.start()]

        title = raw_title.strip()

    title_japanese: str | None = None
    title_romaji: str | None = None

    if subtitle_text:
        # Pattern: "Romaji Title (Japanese Title)"
        jp_m = re.search(r"\(([^)]+)\)\s*$", subtitle_text)
        if jp_m:
            title_japanese = jp_m.group(1).strip()
            title_romaji = subtitle_text[: jp_m.start()].strip()
        else:
            title_romaji = subtitle_text.strip()

    return title, title_japanese, title_romaji, is_filler, is_recap


def _parse_episode_characters(
    raw_items: list[dict[str, Any]] | None,
) -> list[EpisodeCharacterRef]:
    if not raw_items:
        return []

    characters: list[EpisodeCharacterRef] = []
    for item in raw_items:
        name = (item.get("char_name") or "").strip()
        url = (item.get("char_url") or "").strip()
        if not name or not url:
            continue

        # Extract ID from potentially dirty URL (e.g. ".../character/40/Name")
        char_id_match = re.search(r"/character/(\d+)", url)
        if not char_id_match:
            continue
        char_id = int(char_id_match.group(1))

        role = item.get("role") or "Supporting"

        voice_actors_html = item.get("voice_actors_html") or ""
        voice_actors: list[EpisodeVARef] = []
        if voice_actors_html:
            # Match each VA link and the following language text node.
            # Example: <a href=".../people/75/...">Tanaka, Mayumi</a> (Japanese)<br>
            va_matches = re.finditer(
                r'<a[^>]+href="[^"]*/people/(\d+)[^"]*"[^>]*>(.*?)</a>\s*\(([^)]+)\)',
                voice_actors_html,
                re.DOTALL
            )
            for m in va_matches:
                person_id = int(m.group(1))
                va_name = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                language = m.group(3).strip()
                voice_actors.append(
                    EpisodeVARef(
                        person_id=person_id,
                        name=va_name,
                        language=language,
                    )
                )

        characters.append(
            EpisodeCharacterRef(
                mal_id=char_id, name=name, role=role, voice_actors=voice_actors
            )
        )

    return characters


def _parse_episode_staff(raw_items: list[dict[str, Any]] | None) -> list[EpisodeStaffRef]:
    if not raw_items:
        return []

    staff: list[EpisodeStaffRef] = []
    for item in raw_items:
        name = (item.get("name") or "").strip()
        url = (item.get("person_url") or "").strip()
        role = (item.get("role") or "").strip()

        if not name or not url or not role:
            continue

        person_match = re.search(r"/people/(\d+)", url)
        if not person_match:
            continue
        person_id = int(person_match.group(1))

        staff.append(EpisodeStaffRef(person_id=person_id, name=name, role=role))

    return staff


def _normalize_episode_url(identifier: str) -> str:
    """Normalize a MAL episode identifier to a full URL.

    Accepts:
        - Full URL:  "https://myanimelist.net/anime/21/One_Piece/episode/1"
        - With slash: "/anime/21/One_Piece/episode/1"
        - Without:   "anime/21/One_Piece/episode/1"
    """
    if identifier.startswith("http"):
        return identifier
    path = identifier.lstrip("/")
    return f"{MAL_BASE_URL}/{path}"


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_episode_detail",
    dependencies=[
        _get_episode_schema,
        get_mal_docker_browser_config,
        get_mal_docker_crawler_config,
        _parse_episode_characters,
        _parse_episode_staff,
    ],
)
async def _fetch_mal_episode_data(url: str) -> dict[str, Any] | None:
    """Fetch a MAL episode page by full URL and extract data. Cached by URL."""
    from enrichment.crawlers.crawl4ai_docker import crawl_single_url

    await _limiter.acquire()
    result = await crawl_single_url(
        url=url,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(_get_episode_schema()),
    )
    if not result:
        return None

    status = result.get("status_code")
    if status == 404:
        logger.warning(f"Episode not found: {url}")
        return None
    if status and status != 200:
        logger.error(f"HTTP {status} for {url}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    if not raw_list:
        return None
    raw = raw_list[0]
    raw["_url"] = url
    return raw


def _build_episode_from_raw(
    raw: dict[str, Any], episode_number: int, url: str
) -> MalScrapedEpisode:
    """Construct a MalScrapedEpisode from a raw extraction dict."""
    saved_url = raw.pop("_url", url)

    title = raw.get("title") or f"Episode {episode_number}"
    is_filler = bool(raw.get("is_filler"))
    is_recap = bool(raw.get("is_recap"))
    title_japanese = raw.get("title_japanese")
    title_romaji = raw.get("title_romaji")
    aired = parse_iso_date(raw.get("aired_raw"))
    duration = parse_duration_seconds(raw.get("duration_raw"))
    syn_raw = raw.get("synopsis_raw")
    synopsis = " ".join(syn_raw.split()).strip() if syn_raw else None
    characters = _parse_episode_characters(raw.get("characters"))
    staff = _parse_episode_staff(raw.get("staff"))

    return MalScrapedEpisode(
        episode_number=episode_number,
        url=saved_url,
        title=title,
        title_japanese=title_japanese,
        title_romaji=title_romaji,
        synopsis=synopsis,
        aired=aired,
        duration=duration,
        filler=is_filler,
        recap=is_recap,
        characters=characters,
        staff=staff,
    )


async def fetch_mal_episode(
    identifier: str,
    output_path: str | None = None,
) -> MalScrapedEpisode | None:
    """Fetch a MAL episode detail page.

    Args:
        identifier: Full URL, path with leading slash, or path without —
            e.g. "https://myanimelist.net/anime/21/One_Piece/episode/1"
            or   "anime/21/One_Piece/episode/1"
        output_path: Optional path to write JSON output.

    Returns:
        MalScrapedEpisode if successful, None otherwise.
    """
    url = _normalize_episode_url(identifier)

    m = re.search(r"/episode/(\d+)", url)
    if not m:
        logger.error(f"Cannot parse mal_id/episode_number from URL: {url}")
        return None
    episode_number = int(m.group(1))
    logger.info(f"[episode] Fetching episode {episode_number}: {url}")
    raw = await _fetch_mal_episode_data(url)
    if not raw:
        logger.error(f"[episode] Failed to fetch episode {episode_number}")
        return None

    ep = _build_episode_from_raw(raw, episode_number, url)

    if output_path:
        from enrichment.mappers.mal_mapper import episode_from_mal
        safe_path = sanitize_output_path(output_path)
        canonical = episode_from_mal(ep)
        Path(safe_path).write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return ep


async def fetch_mal_episodes(
    identifiers: list[str],
    *,
    output_path: str | None = None,
) -> list[MalScrapedEpisode | None]:
    """Fetch multiple MAL episode pages as a single Docker batch job.

    Submits all URLs in one POST /crawl/job request and polls a single task ID —
    O(1) polling tasks regardless of episode count.  Suitable for bulk first-time
    fetches (e.g. all episodes of an anime during enrichment).

    For single-episode fetches with Redis caching, use ``fetch_mal_episode`` instead.

    Args:
        identifiers: List of episode URLs or paths (same formats as fetch_mal_episode).
        output_path: If provided, each successfully parsed episode is appended to this
            JSONL file as it is processed (streaming write).

    Returns:
        List aligned to ``identifiers`` — None for any failed fetch.
    """
    from enrichment.crawlers.crawl4ai_docker import crawl_batch_urls

    urls = [_normalize_episode_url(i) for i in identifiers]
    logger.info(f"[episodes] Fetching {len(urls)} episodes...")
    docker_results = await crawl_batch_urls(
        urls,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(_get_episode_schema()),
    )

    episodes: list[MalScrapedEpisode | None] = []
    for url, result in zip(urls, docker_results):
        if not result:
            episodes.append(None)
            continue

        status = result.get("status_code")
        if status == 404:
            logger.warning("Episode not found: %s", url)
            episodes.append(None)
            continue
        if status and status != 200:
            logger.error("HTTP %s for %s", status, url)
            episodes.append(None)
            continue

        raw_list = json.loads(result.get("extracted_content") or "[]")
        if not raw_list:
            episodes.append(None)
            continue

        raw = raw_list[0]
        raw["_url"] = url

        m = re.search(r"/episode/(\d+)", url)
        if not m:
            logger.error("Cannot parse episode_number from URL: %s", url)
            episodes.append(None)
            continue

        ep = _build_episode_from_raw(raw, int(m.group(1)), url)
        if output_path is not None:
            from enrichment.mappers.mal_mapper import episode_from_mal
            _append_jsonl(output_path, episode_from_mal(ep))
        episodes.append(ep)

    return episodes


async def main() -> int:
    """CLI entry point for fetching a MAL episode page."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Fetch episode data from MAL")
    parser.add_argument(
        "identifier",
        type=str,
        help="Episode URL or path, e.g. 'anime/21/One_Piece/episode/1' or full URL",
    )
    parser.add_argument("--output", type=str, default="mal_episode.json")
    args = parser.parse_args()

    ep = await fetch_mal_episode(args.identifier, output_path=args.output)
    if ep is None:
        logger.error(f"Failed to fetch or parse episode data for: {args.identifier}")
        return 1

    logger.info(
        f"Fetched Episode {ep.episode_number}: {ep.title}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
