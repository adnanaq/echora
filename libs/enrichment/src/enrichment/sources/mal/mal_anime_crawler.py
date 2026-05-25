"""MAL Anime Detail Crawler.

Fetches /anime/{id} and /anime/{id}/pics pages from MyAnimeList and extracts
all anime metadata into a MalAnime model.

Replaces the Jikan API calls in MalClient — one page fetch instead of one API
call per field. No rate-limit queue on Jikan's side.

Usage:
    from enrichment.sources.mal.mal_anime_crawler import fetch_mal_anime

    anime = await fetch_mal_anime(21)          # One Piece
    anime = await fetch_mal_anime(21, output_path="one_piece.json")
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Any

from enrichment.sources.base.crawl4ai_docker import crawl_batch_urls
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
from enrichment.sources.base.framework import (
    BaseCrawler,
    DockerTransport,
    FileRepository,
    NullRepository,
)
from enrichment.sources.mal.mal_base import (
    MAL_BASE_URL,
    get_mal_scraping_limiter,
    parse_aired_string,
    parse_broadcast_string,
    parse_duration_seconds,
    parse_episode_ranges,
    parse_number,
    parse_premiered,
)
from enrichment.sources.mal.mal_mapper import anime_from_mal
from enrichment.sources.mal.mal_models import (
    MalAnime,
    MalCompanyRef,
    MalEpisodeRange,
    MalExternalLink,
    MalRelatedEntry,
    MalThemeSong,
    MalTrailer,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan  # Reuse Jikan TTL for MAL scraping (24h default)

_limiter = get_mal_scraping_limiter()


def _get_anime_schema() -> dict[str, Any]:
    """XPath extraction schema for MAL anime detail pages.

    Uses schema.org attributes for the most stable selectors. The sidebar
    is extracted as a single HTML block for post-processing via sidebar parsers.
    """
    return {
        "name": "MalAnimeDetail",
        "baseSelector": "//body",
        "fields": [
            # Basic Sidebar Fields (extracted via regex from their containing divs)
            {
                "name": "type",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Type:')]]",
                "type": "regex",
                "pattern": r"Type:\s*(.*)",
            },
            {
                "name": "episodes",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Episodes:')]]",
                "type": "regex",
                "pattern": r"Episodes:\s*(.*)",
            },
            {
                "name": "status",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Status:')]]",
                "type": "regex",
                "pattern": r"Status:\s*(.*)",
            },
            {
                "name": "duration_raw",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Duration:')]]",
                "type": "regex",
                "pattern": r"Duration:\s*(.*)",
            },
            {
                "name": "source_material",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Source:')]]",
                "type": "regex",
                "pattern": r"Source:\s*(.*)",
            },
            {
                "name": "rating",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Rating:')]]",
                "type": "regex",
                "pattern": r"Rating:\s*(.*)",
            },
            {
                "name": "aired_raw",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Aired:')]]",
                "type": "regex",
                "pattern": r"Aired:\s*(.*)",
            },
            {
                "name": "premiered_raw",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Premiered:')]]",
                "type": "regex",
                "pattern": r"Premiered:\s*(.*)",
            },
            {
                "name": "broadcast_raw",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Broadcast:')]]",
                "type": "regex",
                "pattern": r"Broadcast:\s*(.*)",
            },
            # Alternative Titles
            {
                "name": "title_english",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'English:')]]",
                "type": "regex",
                "pattern": r"English:\s*(.*)",
            },
            {
                "name": "title_japanese",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Japanese:')]]",
                "type": "regex",
                "pattern": r"Japanese:\s*(.*)",
            },
            {
                "name": "synonyms_raw",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Synonyms:')]]",
                "type": "regex",
                "pattern": r"Synonyms:\s*(.*)",
            },
            # Statistics
            {
                "name": "rank_html",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Ranked:')]]",
                "type": "html",
            },
            {
                "name": "popularity",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Popularity:')]]",
                "type": "regex",
                "pattern": r"Popularity:\s*#?(\d+)",
            },
            {
                "name": "members",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Members:')]]",
                "type": "regex",
                "pattern": r"Members:\s*([\d,]+)",
            },
            {
                "name": "favorites",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Favorites:')]]",
                "type": "regex",
                "pattern": r"Favorites:\s*([\d,]+)",
            },
            # Arrays
            {
                "name": "genres",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Genre')]]/a",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "themes",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Theme')]]/a",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "demographics",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Demographic')]]/a",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            # Companies
            {
                "name": "producers",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Producers')]]/a",
                "type": "list",
                "fields": [
                    {"name": "name", "selector": ".", "type": "text"},
                    {
                        "name": "source",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
            {
                "name": "licensors",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Licensors')]]/a",
                "type": "list",
                "fields": [
                    {"name": "name", "selector": ".", "type": "text"},
                    {
                        "name": "source",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
            {
                "name": "studios",
                "selector": "//div[span[contains(@class,'dark_text')][contains(.,'Studios')]]/a",
                "type": "list",
                "fields": [
                    {"name": "name", "selector": ".", "type": "text"},
                    {
                        "name": "source",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
            # Background
            {
                "name": "background_raw",
                "selector": "//h2[@id='background']/parent::div/parent::td",
                "type": "html",
            },
            # Related entries
            {
                "name": "related_tile_entries",
                "selector": "//div[contains(@class,'entries-tile')]/div[contains(@class,'entry')]",
                "type": "list",
                "fields": [
                    {
                        "name": "relation_raw",
                        "selector": ".//div[contains(@class,'relation')]",
                        "type": "text",
                    },
                    {
                        "name": "title",
                        "selector": ".//div[contains(@class,'title')]/a",
                        "type": "regex",
                        "pattern": r"^(.*?)(?:\s*\([^)]+\))?\s*$",
                    },
                    {
                        "name": "entry_type",
                        "selector": ".//div[contains(@class,'title')]",
                        "type": "regex",
                        "pattern": r"\(([^)]+)\)\s*$",
                    },
                    {
                        "name": "source",
                        "selector": ".//div[contains(@class,'title')]/a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
            {
                "name": "related_table_entries",
                "selector": "//table[contains(@class,'entries-table')]//tr[td[2]]",
                "type": "list",
                "fields": [
                    {
                        "name": "relation",
                        "selector": "./td[1]",
                        "type": "text",
                    },
                    {
                        "name": "links_html",
                        "selector": "./td[2]",
                        "type": "html",
                    },
                ],
            },
            # Links
            {
                "name": "external_sources_raw",
                "selector": (
                    "//h2[normalize-space()='Available At' or normalize-space()='Resources']"
                    "/following-sibling::div[1][contains(@class,'external_links')]"
                    "//a[@href and not(@href='#')]"
                ),
                "type": "list",
                "fields": [
                    {
                        "name": "name",
                        "selector": ".//div[contains(@class,'caption')]",
                        "type": "text",
                    },
                    {
                        "name": "source",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
            {
                "name": "streaming_links_raw",
                "selector": (
                    "//h2[normalize-space()='Streaming Platforms']"
                    "/following-sibling::div[1][contains(@class,'broadcasts')]"
                    "//a[@href and @title]"
                ),
                "type": "list",
                "fields": [
                    {
                        "name": "name",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "title",
                    },
                    {
                        "name": "source",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
            # Titles
            {
                "name": "title",
                "selector": "//h1[contains(@class,'title-name')]/strong",
                "type": "text",
            },
            {
                "name": "title_og",
                "selector": "//meta[@property='og:title']",
                "type": "attribute",
                "attribute": "content",
            },
            # Statistics (schema.org — extremely stable)
            {
                "name": "score",
                "selector": "//span[@itemprop='ratingValue']",
                "type": "text",
            },
            {
                "name": "scored_by",
                "selector": "//span[@itemprop='ratingCount']",
                "type": "text",
            },
            # Description (schema.org)
            {
                "name": "synopsis",
                "selector": "//p[@itemprop='description']",
                "type": "text",
            },
            # Cover image (schema.org)
            {
                "name": "cover_image_src",
                "selector": "//img[@itemprop='image']",
                "type": "attribute",
                "attribute": "data-src",
            },
            # Theme songs
            {
                "name": "opening_themes_raw",
                "selector": "//div[contains(@class,'theme-songs') and contains(@class,'opnening')]//tr[td[2]]",
                "type": "list",
                "fields": [
                    {"name": "title_text", "selector": ".//td[2]", "type": "text"},
                    {
                        "name": "artist",
                        "selector": ".//span[contains(@class,'theme-song-artist')]",
                        "type": "text",
                    },
                    {
                        "name": "episodes",
                        "selector": ".//span[contains(@class,'theme-song-episode')]",
                        "type": "text",
                    },
                ],
            },
            {
                "name": "ending_themes_raw",
                "selector": "//div[contains(@class,'theme-songs') and contains(@class,'ending')]//tr[td[2]]",
                "type": "list",
                "fields": [
                    {"name": "title_text", "selector": ".//td[2]", "type": "text"},
                    {
                        "name": "artist",
                        "selector": ".//span[contains(@class,'theme-song-artist')]",
                        "type": "text",
                    },
                    {
                        "name": "episodes",
                        "selector": ".//span[contains(@class,'theme-song-episode')]",
                        "type": "text",
                    },
                ],
            },
            # Trailer
            {
                "name": "trailer_embed_url",
                "selector": "//div[contains(@class,'video-promotion')]//a",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "trailer_title",
                "selector": "//div[contains(@class,'video-promotion')]//span[contains(@class,'title')]",
                "type": "text",
            },
            # Background (text node after h2 "Background")
            {
                "name": "background_html",
                "selector": "//td[contains(@class,'pb16')]//p",
                "type": "html",
            },
            # Gallery pictures — picSurround <a href> contains full-size 'l' image URL.
            # Returns [] on the anime detail page (no picSurround divs there);
            # returns all gallery URLs when this schema is applied to the /pics page.
            {
                "name": "picture_urls_raw",
                "selector": "//div[contains(@class,'picSurround')]/a[@href]",
                "type": "list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
        ],
    }


def _parse_trailer(raw: dict[str, Any]) -> MalTrailer | None:
    """Build a MalTrailer from the raw video-promotion fields.

    Converts the YouTube nocookie embed URL to a watchable URL and derives
    the thumbnail from the video ID.
    """
    embed_url = raw.get("trailer_embed_url") or ""
    m = re.search(r"embed/([^?&]+)", embed_url)
    if not m:
        return None
    vid = m.group(1)
    return MalTrailer(
        source=f"https://www.youtube.com/watch?v={vid}",
        title=raw.get("trailer_title"),
        thumbnail=f"https://img.youtube.com/vi/{vid}/maxresdefault.jpg",
    )


def _parse_picture_urls(pics_data: list[dict[str, Any]]) -> list[str]:
    """Extract large picture URLs from the crawled pics page data."""
    urls = []
    for item in pics_data:
        url = item.get("url") or ""
        if url and "myanimelist" in url and "images/anime" in url:
            urls.append(url)
    return urls


def _normalize_mal_url(path: str) -> str:
    """Ensure a MAL path is a full URL."""
    if not path:
        return ""
    if path.startswith("http"):
        return path
    return f"{MAL_BASE_URL}{path if path.startswith('/') else '/' + path}"


def _parse_structured_themes(raw_themes: list[dict[str, Any]]) -> list[MalThemeSong]:
    """Process structured theme song lists from the schema."""
    results = []
    for raw in raw_themes:
        title_text = (raw.get("title_text") or "").strip()

        # Skip marketing junk (Spotify, etc.) - they don't have quoted titles
        if '"' not in title_text:
            continue

        # Extract title from double quotes (MAL standard)
        title_match = re.search(r'"([^"]+)"', title_text)
        title = title_match.group(1) if title_match else title_text

        if not title:
            continue

        # Clean artist (remove leading ' by ')
        artist = (raw.get("artist") or "").strip()
        artist = re.sub(r"^by\s+", "", artist, flags=re.IGNORECASE).strip()

        # Parse episode ranges
        episodes_raw = raw.get("episodes")
        episode_ranges = [
            MalEpisodeRange(start=s, end=e)
            for s, e in parse_episode_ranges(episodes_raw)
        ]

        results.append(
            MalThemeSong(title=title, artist=artist or None, episodes=episode_ranges)
        )
    return results


def _parse_all_related_entries(raw: dict[str, Any]) -> list[MalRelatedEntry]:
    """Process both tile and table related entries from raw schema data."""
    unified_items = []

    # 1. Gather from Tile format (e.g. Adaptations)
    for entry in raw.get("related_tile_entries", []):
        raw_rel = (entry.get("relation_raw") or "").strip()
        # Handle "Adaptation\n(Manga)" structure in tiles
        rel_parts = [p.strip() for p in raw_rel.split("\n") if p.strip()]
        relation = rel_parts[0] if rel_parts else ""

        entry_type = entry.get("entry_type")
        if not entry_type and len(rel_parts) > 1:
            type_match = re.search(r"\(([^)]+)\)", rel_parts[1])
            if type_match:
                entry_type = type_match.group(1)

        unified_items.append(
            {
                "relation": relation,
                "title": entry.get("title"),
                "entry_type": entry_type,
                "source": entry.get("source"),
            }
        )

    # 2. Gather from Table format (e.g. Side Stories)
    for row in raw.get("related_table_entries", []):
        raw_rel = (row.get("relation") or "").strip()
        # Clean relation strings
        rel_parts = [p.strip() for p in raw_rel.split("\n") if p.strip()]
        relation = rel_parts[0] if rel_parts else ""

        links_html = row.get("links_html") or ""
        # Match each link and the following text node (potentially containing format)
        # Example: <li><a href="...">Title</a> (TV)</li>
        for match in re.finditer(
            r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?:</li>|<li>|<a|</ul>|$)',
            links_html,
            re.DOTALL,
        ):
            source_url = match.group(1).strip()
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            format_text = match.group(3).strip()

            # Extract type from format_text (e.g. "(TV)")
            entry_type = None
            type_match = re.search(r"\(([^)]+)\)", format_text)
            if type_match:
                entry_type = type_match.group(1)
            elif len(rel_parts) > 1:
                # Fallback to type in relation cell
                type_match = re.search(r"\(([^)]+)\)", rel_parts[1])
                if type_match:
                    entry_type = type_match.group(1)

            unified_items.append(
                {
                    "relation": relation,
                    "title": title,
                    "entry_type": entry_type,
                    "source": source_url,
                }
            )

    # 3. Single processing loop
    related_entries = []
    for item in unified_items:
        relation = (item.get("relation") or "").strip().rstrip(":")
        source_url = _normalize_mal_url(item.get("source", ""))
        title = (item.get("title") or "").strip()

        if not title or not source_url:
            continue

        related_entries.append(
            MalRelatedEntry(
                relation=relation,
                title=title,
                source=source_url,
                entry_type=item.get("entry_type"),
                is_anime="/anime/" in source_url,
            )
        )

    return related_entries


def _build_anime_from_raw(
    raw: dict[str, Any],
    url: str,
    picture_urls: list[str],
) -> MalAnime:
    """Transform raw extracted data into a MalAnime model.

    This is the post-processing step that converts CSS-extracted strings into
    typed/parsed values using the sidebar parser utilities.

    Args:
        raw: Dict from JsonCssExtractionStrategy output.
        url: Canonical URL for this anime.
        picture_urls: Gallery image URLs from /anime/{id}/pics.

    Returns:
        Validated MalAnime model.
    """
    # Titles
    title = (raw.get("title") or raw.get("title_og") or "").strip()
    title_english = raw.get("title_english")
    title_japanese = raw.get("title_japanese")
    synonyms_str = raw.get("synonyms_raw")
    synonyms = [s.strip() for s in synonyms_str.split(",")] if synonyms_str else []

    # Classification
    anime_type = raw.get("type")
    status = raw.get("status")
    source_material = raw.get("source_material")
    rating = raw.get("rating")

    # Episode count
    ep_count_raw = raw.get("episodes")
    episode_count: int | None = None
    if ep_count_raw and ep_count_raw.lower() not in ("unknown", "?", "n/a"):
        try:
            episode_count = int(ep_count_raw.replace(",", ""))
        except ValueError:
            episode_count = None

    # Duration
    duration_raw = raw.get("duration_raw")
    duration = parse_duration_seconds(duration_raw) if duration_raw else None

    # Aired dates
    aired_raw = raw.get("aired_raw")
    aired_from, aired_to = parse_aired_string(aired_raw)

    # Season / year
    premiered_raw = raw.get("premiered_raw")
    season, year = parse_premiered(premiered_raw)

    # Broadcast
    broadcast_raw = raw.get("broadcast_raw")
    broadcast_day, broadcast_time, broadcast_timezone = parse_broadcast_string(
        broadcast_raw
    )

    # Statistics
    score_val = raw.get("score")
    score = float(score_val.strip()) if score_val and score_val.strip() else None

    scored_by = parse_number(raw.get("scored_by"))

    rank_html = raw.get("rank_html")
    rank: int | None = None
    if rank_html:
        # Match only the digits following '#' before any nested tag like <sup>
        rank_match = re.search(r"#(\d+)", rank_html)
        if rank_match:
            rank = int(rank_match.group(1))

    popularity = parse_number(raw.get("popularity"))
    members = parse_number(raw.get("members"))
    favorites = parse_number(raw.get("favorites"))

    # Arrays
    genres = [g["name"] for g in raw.get("genres", [])]
    themes = [t["name"] for t in raw.get("themes", [])]
    demographics = [d["name"] for d in raw.get("demographics", [])]

    # Companies
    producers = [
        MalCompanyRef(
            name=item["name"].strip(), source=_normalize_mal_url(item["source"])
        )
        for item in raw.get("producers", [])
        if item.get("name")
        and item.get("source")
        and "dbchanges.php" not in item["source"]
    ]
    licensors = [
        MalCompanyRef(
            name=item["name"].strip(), source=_normalize_mal_url(item["source"])
        )
        for item in raw.get("licensors", [])
        if item.get("name")
        and item.get("source")
        and "dbchanges.php" not in item["source"]
    ]
    studios = [
        MalCompanyRef(
            name=item["name"].strip(), source=_normalize_mal_url(item["source"])
        )
        for item in raw.get("studios", [])
        if item.get("name")
        and item.get("source")
        and "dbchanges.php" not in item["source"]
    ]

    # Theme songs
    opening_themes = _parse_structured_themes(raw.get("opening_themes_raw", []))
    ending_themes = _parse_structured_themes(raw.get("ending_themes_raw", []))

    # Images — cover (large variant) first, then gallery; deduplicated via dict.fromkeys
    cover_url = raw.get("cover_image_src") or ""
    if cover_url and cover_url.endswith(".jpg") and not cover_url.endswith("l.jpg"):
        cover_url = cover_url[:-4] + "l.jpg"
    picture_urls = list(
        dict.fromkeys(([cover_url] if cover_url else []) + picture_urls)
    )
    images: dict[str, str] = {}

    # External links
    external_sources = [
        MalExternalLink(name=item["name"].strip(), source=item["source"])
        for item in raw.get("external_sources_raw", [])
        if item.get("name") and item.get("source")
    ]

    # Streaming links
    streaming = [
        MalExternalLink(name=item["name"].strip(), source=item["source"])
        for item in raw.get("streaming_links_raw", [])
        if item.get("name") and item.get("source")
    ]

    # Background — strip placeholder text
    bg_html = raw.get("background_raw") or ""
    background: str | None = None
    if bg_html:
        # Extract text after the background ID header but before the next section div or end
        bg_match = re.search(
            r'id="background".*?</h2>(.*?)(?:<div|<!--|$)', bg_html, re.DOTALL
        )
        if bg_match:
            bg_text = re.sub(r"<[^>]+>", "", bg_match.group(1)).strip()
            if bg_text and "no background information" not in bg_text.lower():
                background = bg_text

    # Relations
    related_entries = _parse_all_related_entries(raw)

    return MalAnime(
        source=url,
        title=title,
        title_english=title_english,
        title_japanese=title_japanese,
        synonyms=synonyms,
        type=anime_type,
        status=status,
        source_material=source_material,
        rating=rating,
        year=year,
        season=season,
        aired_from=aired_from,
        aired_to=aired_to,
        broadcast_day=broadcast_day,
        broadcast_time=broadcast_time,
        broadcast_timezone=broadcast_timezone,
        episode_count=episode_count,
        duration=duration,
        score=score,
        scored_by=scored_by,
        rank=rank,
        popularity=popularity,
        members=members,
        favorites=favorites,
        synopsis=raw.get("synopsis"),
        background=background,
        genres=genres,
        themes=themes,
        demographics=demographics,
        producers=producers,
        licensors=licensors,
        studios=studios,
        related_entries=related_entries,
        images=images,
        picture_urls=picture_urls,
        trailer=_parse_trailer(raw),
        opening_themes=opening_themes,
        ending_themes=ending_themes,
        external_sources=external_sources,
        streaming=streaming,
    )


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_anime_scraped",
    dependencies=[_get_anime_schema, _parse_picture_urls],
)
async def _fetch_mal_anime_data(url: str) -> dict[str, Any] | None:
    """Fetch and extract MAL anime detail page via crawl4ai Docker. Cached by URL.

    Accepts both slugged and slugless URLs. If the URL already contains a slug
    (e.g. /anime/21/One_Piece), slug derivation is skipped. Otherwise the slug
    is derived from episode links on the page before fetching /pics.

    Cache key includes schema hash for automatic invalidation when extraction
    logic changes.

    Args:
        url: MAL anime URL — slugged or slugless.

    Returns:
        Raw extracted data dict, or None on failure.
    """
    # ── Step 1: fetch anime detail page ──────────────────────────────────────
    await _limiter.acquire()
    browser_config = get_docker_browser_config()
    crawler_config = get_docker_crawler_config(_get_anime_schema(), delay=1.5)

    results = await crawl_batch_urls(
        [url], browser_config=browser_config, crawler_config=crawler_config
    )
    result = results[0] if results else None

    if not result:
        logger.warning(f"No result for anime page {url}")
        return None

    status = result.get("status_code")
    if status == 404:
        logger.warning(f"Anime not found (404): {url}")
        return None
    if status and status != 200:
        logger.error(f"HTTP {status} for anime {url}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    if not raw_list:
        logger.warning(f"No data extracted from anime page {url}")
        return None
    raw = raw_list[0]

    canonical_url = result.get("metadata", {}).get("og:url") or result.get("url") or url

    # ── Step 2: fetch /pics page using correct slug URL ───────────────────────
    # /anime/{id}/pics (no slug) returns the wrong page on MAL — slug is required.
    picture_urls: list[str] = []
    pics_url = f"{canonical_url}/pics"
    await _limiter.acquire()
    pics_results = await crawl_batch_urls(
        [pics_url], browser_config=browser_config, crawler_config=crawler_config
    )
    pics_result = pics_results[0] if pics_results else None
    if pics_result:
        pics_list = json.loads(pics_result.get("extracted_content") or "[]")
        if pics_list:
            picture_urls = _parse_picture_urls(pics_list[0].get("picture_urls_raw", []))

    raw["_picture_urls"] = picture_urls
    raw["_url"] = canonical_url
    return raw


class MalAnimeCrawler(BaseCrawler[MalAnime, dict[str, Any]]):
    """Crawler for MyAnimeList anime detail pages."""

    def get_extraction_schema(self) -> dict[str, Any]:
        return _get_anime_schema()

    def normalize_identifier(self, identifier: str) -> str:
        return _normalize_mal_url(identifier)

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_mal_anime_data(url)

    def build_source_model(self, processed_raw: dict[str, Any], url: str) -> MalAnime:
        picture_urls = processed_raw.pop("_picture_urls", [])
        saved_url = processed_raw.pop("_url", url)
        return _build_anime_from_raw(processed_raw, saved_url, picture_urls)

    def map_to_canonical(self, source_model: MalAnime) -> dict[str, Any]:
        return anime_from_mal(source_model)


async def fetch_mal_anime(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch MAL anime detail and return canonical anime dict.

    Args:
        url: MAL anime URL (slugged or slugless).
        output_path: If provided, write the canonical dict to this JSON file.

    Returns:
        Canonical anime dict, or None on failure.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await MalAnimeCrawler(DockerTransport(), repo).crawl(url)


async def main() -> int:
    """CLI entry point for fetching a MAL anime page."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fetch anime data from MAL")
    parser.add_argument(
        "url", type=str, help="MAL anime URL (e.g., https://myanimelist.net/anime/21)"
    )
    parser.add_argument(
        "--output", type=str, default="mal_anime.json", help="Output file path"
    )
    args = parser.parse_args()

    anime_dict = await fetch_mal_anime(args.url, output_path=args.output)
    if anime_dict is None:
        logger.error(f"No data extracted for anime URL {args.url}")
        return 1
    logger.info(
        f"Done: {anime_dict.get('title')} ({anime_dict.get('episode_count')} episodes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
