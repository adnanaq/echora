"""
This script crawls anime information from anime-planet.com anime pages.

It accepts a slug as a command-line argument and extracts comprehensive anime data
including related anime, rankings, studios, and all metadata from JSON-LD.

The extracted data is saved to 'animeplanet_anime.json' in the project root.

Usage:
    python anime_planet_anime_crawler.py <slug>
    python anime_planet_anime_crawler.py dandadan
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, cast

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from src.cache_manager.config import get_cache_config
from src.cache_manager.result_cache import cached_result
from src.enrichment.crawlers.utils import sanitize_output_path

# Get TTL from config to keep cache control centralized
_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_ANIME_URL = "https://www.anime-planet.com/anime/"


def _normalize_anime_url(anime_identifier: str) -> str:
    """
    Normalize various input formats to full anime-planet URL.

    Accepts:
        - Full URL: "https://www.anime-planet.com/anime/dandadan"
        - Slug: "dandadan"
        - Path: "/anime/dandadan"

    Returns:
        Full URL: "https://www.anime-planet.com/anime/dandadan"
    """
    if not anime_identifier.startswith("http"):
        # Remove leading slashes and "anime/" prefix if present
        clean_id = anime_identifier.lstrip("/")
        if clean_id.startswith("anime/"):
            clean_id = clean_id[6:]  # Remove "anime/" prefix
        url = f"{BASE_ANIME_URL}{clean_id}"
    else:
        url = anime_identifier

    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(
            f"Invalid URL: Must be an anime-planet.com anime URL. "
            f"Expected format: '{BASE_ANIME_URL}<slug>' or just '<slug>'"
        )

    return url


def _extract_slug_from_url(url: str) -> str:
    """Extract slug from anime-planet URL."""
    # Extract slug from: https://www.anime-planet.com/anime/dandadan
    match = re.search(r"/anime/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Could not extract slug from URL: {url}")
    return match.group(1)


@cached_result(ttl=TTL_ANIME_PLANET, key_prefix="animeplanet_anime")
async def _fetch_animeplanet_anime_data(canonical_slug: str) -> Optional[Dict[str, Any]]:
    """
    Pure cached function that crawls and processes anime data from anime-planet.com.
    All data is available on the main anime page - no navigation needed.

    Results are cached in Redis for 24 hours based ONLY on canonical slug.
    This function has no side effects - it only fetches and returns data.

    Args:
        canonical_slug: Canonical anime slug (e.g., "dandadan") - already normalized by caller

    Returns:
        Complete anime data dictionary, or None on failure
    """
    # Build URL from canonical slug (caller already normalized)
    url = f"{BASE_ANIME_URL}{canonical_slug}"

    css_schema = {
        "baseSelector": "body",
        "fields": [
            # Related anime from same franchise section
            {
                "name": "related_anime_raw",
                "selector": "#tabs--relations--anime--same_franchise .pure-u-1",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "title",
                        "selector": ".RelatedEntry__name",
                        "type": "text",
                    },
                    {
                        "name": "url",
                        "selector": "a.RelatedEntry",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "relation_subtype",
                        "selector": ".RelatedEntry__subtitle",
                        "type": "text",
                    },
                    {
                        "name": "image",
                        "selector": ".RelatedEntry__image",
                        "type": "attribute",
                        "attribute": "src",
                    },
                    {
                        "name": "start_date_attr",
                        "selector": "time span[data-start-date]",
                        "type": "attribute",
                        "attribute": "data-start-date",
                    },
                    {
                        "name": "end_date_attr",
                        "selector": "time span[data-end-date]",
                        "type": "attribute",
                        "attribute": "data-end-date",
                    },
                    {
                        "name": "metadata_text",
                        "selector": ".RelatedEntry__metadata",
                        "type": "text",
                    },
                ],
            },
            # Related manga from same franchise section
            {
                "name": "related_manga_raw",
                "selector": "#tabs--relations--manga .pure-u-1",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "title",
                        "selector": ".RelatedEntry__name",
                        "type": "text",
                    },
                    {
                        "name": "url",
                        "selector": "a.RelatedEntry",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "relation_subtype",
                        "selector": ".RelatedEntry__subtitle",
                        "type": "text",
                    },
                    {
                        "name": "image",
                        "selector": ".RelatedEntry__image",
                        "type": "attribute",
                        "attribute": "src",
                    },
                    {
                        "name": "start_date_attr",
                        "selector": "time span[data-start-date]",
                        "type": "attribute",
                        "attribute": "data-start-date",
                    },
                    {
                        "name": "end_date_attr",
                        "selector": "time span[data-end-date]",
                        "type": "attribute",
                        "attribute": "data-end-date",
                    },
                    {
                        "name": "metadata_text",
                        "selector": ".RelatedEntry__metadata",
                        "type": "text",
                    },
                ],
            },
            # Rank from entry bar
            {
                "name": "rank_text",
                "selector": ".entryBar .pure-1",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            # Studios
            {
                "name": "studios_raw",
                "selector": "a[href*='/studios/']",
                "type": "list",
                "fields": [{"name": "studio", "type": "text"}],
            },
            # Japanese title
            {
                "name": "title_japanese",
                "selector": "h2.aka",
                "type": "text",
            },
            # Poster from opengraph (different from JSON-LD image)
            {
                "name": "poster",
                "selector": "meta[property='og:image']",
                "type": "attribute",
                "attribute": "content",
            },
        ],
    }

    async with AsyncWebCrawler() as crawler:
        extraction_strategy = JsonCssExtractionStrategy(css_schema)
        config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

        logging.info(f"Fetching anime data: {url}")
        results: RunManyReturn = await crawler.arun(url=url, config=config)

        if not results:
            logging.warning("No results found.")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if not data:
                    logging.warning("Extraction returned empty data.")
                    return None

                anime_data = cast(Dict[str, Any], data[0])

                # Extract JSON-LD data (most comprehensive source)
                if result.html:
                    json_ld = _extract_json_ld(result.html)
                else:
                    json_ld = None

                if json_ld:
                    # Move @type to parent level
                    if json_ld.get("@type"):
                        anime_data["type"] = json_ld["@type"]

                    # Flatten essential fields to top level
                    if json_ld.get("name"):
                        anime_data["title"] = json_ld["name"]
                    if json_ld.get("description"):
                        anime_data["description"] = json_ld["description"]
                    if json_ld.get("url"):
                        anime_data["url"] = json_ld["url"]
                    if json_ld.get("image"):
                        anime_data["image"] = json_ld["image"]
                    if json_ld.get("startDate"):
                        anime_data["start_date"] = json_ld["startDate"]
                    if json_ld.get("endDate"):
                        anime_data["end_date"] = json_ld["endDate"]
                    if json_ld.get("numberOfEpisodes"):
                        anime_data["episodes"] = json_ld["numberOfEpisodes"]
                    if json_ld.get("genre"):
                        anime_data["genres"] = json_ld["genre"]
                    if json_ld.get("aggregateRating"):
                        anime_data["aggregate_rating"] = json_ld["aggregateRating"]

                # Add slug (passed as canonical_slug parameter)
                anime_data["slug"] = canonical_slug

                # Process rank
                rank = _extract_rank(
                    cast(List[Dict[str, str]], anime_data.get("rank_text", []))
                )
                if rank:
                    anime_data["rank"] = rank
                if "rank_text" in anime_data:
                    del anime_data["rank_text"]

                # Process studios
                studios = _extract_studios(anime_data.get("studios_raw", []))
                if studios:
                    anime_data["studios"] = studios
                if "studios_raw" in anime_data:
                    del anime_data["studios_raw"]

                # Process Japanese title
                if anime_data.get("title_japanese"):
                    title_ja = anime_data["title_japanese"]
                    if title_ja.startswith("Alt title:"):
                        title_ja = title_ja.replace("Alt title:", "").strip()
                    anime_data["title_japanese"] = title_ja

                # Extract poster from og:image if not found by CSS
                if not anime_data.get("poster") and result.html:
                    poster_match = re.search(
                        r'<meta property="og:image" content="([^"]+)"', result.html
                    )
                    if poster_match:
                        anime_data["poster"] = poster_match.group(1)

                # Process related anime
                related_anime = _process_related_anime(
                    anime_data.get("related_anime_raw", [])
                )
                if related_anime:
                    anime_data["related_anime"] = related_anime
                    anime_data["related_count"] = len(related_anime)
                if "related_anime_raw" in anime_data:
                    del anime_data["related_anime_raw"]

                # Process related manga
                related_manga = _process_related_manga(
                    anime_data.get("related_manga_raw", [])
                )
                if related_manga:
                    anime_data["related_manga"] = related_manga
                if "related_manga_raw" in anime_data:
                    del anime_data["related_manga_raw"]

                # Derive year, season, status from dates
                if json_ld and json_ld.get("startDate"):
                    start_date = json_ld["startDate"]
                    end_date = json_ld.get("endDate")

                    # Extract year
                    year_match = re.search(r"(\d{4})", start_date)
                    if year_match:
                        anime_data["year"] = int(year_match.group(1))

                    # Determine season
                    season = _determine_season_from_date(start_date)
                    if season:
                        anime_data["season"] = season

                    # Determine status
                    if start_date and end_date:
                        anime_data["status"] = "COMPLETED"
                    elif start_date and not end_date:
                        from datetime import datetime, timezone

                        try:
                            start_dt = datetime.fromisoformat(
                                start_date.replace("Z", "+00:00")
                            )
                            # If the parsed date is naive, make it aware (assume UTC)
                            if start_dt.tzinfo is None:
                                start_dt = start_dt.replace(tzinfo=timezone.utc)

                            now = datetime.now(timezone.utc)

                            if start_dt > now:
                                anime_data["status"] = "UPCOMING"
                            else:
                                anime_data["status"] = "AIRING"
                        except (ValueError, TypeError):
                            anime_data["status"] = "AIRING"
                else:
                    anime_data["status"] = "UNKNOWN"

                # Return pure data (no side effects)
                return anime_data
            else:
                logging.warning(f"Extraction failed: {result.error_message}")
                return None


async def fetch_animeplanet_anime(
    slug: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Wrapper function that handles side effects (file writing, return_data logic).

    This function calls the cached _fetch_animeplanet_anime_data() to get the data,
    then performs side effects that should execute regardless of cache status.

    Args:
        slug: Anime slug (e.g., "dandadan"), path (e.g., "/anime/dandadan"),
              or full URL (e.g., "https://www.anime-planet.com/anime/dandadan")
        return_data: Whether to return the data dict (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        Complete anime data dictionary (if return_data=True), otherwise None
    """
    # Normalize identifier once so cache keys depend on canonical slug
    # This ensures cache reuse across different identifier formats
    anime_url = _normalize_anime_url(slug)
    canonical_slug = _extract_slug_from_url(anime_url)

    # Fetch data from cache or crawl (pure function keyed only on canonical slug)
    data = await _fetch_animeplanet_anime_data(canonical_slug)

    if data is None:
        return None

    # Side effect: Write to file (always executes, even on cache hit)
    if output_path:
        safe_path = sanitize_output_path(output_path)
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Data written to {safe_path}")

    # Return data based on return_data parameter
    if return_data:
        return data
    return None


def _extract_json_ld(html: str) -> Optional[Dict[str, Any]]:
    """Extract JSON-LD structured data from HTML."""
    try:
        import html as html_lib

        # Find JSON-LD script tag
        match = re.search(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            html,
            re.DOTALL,
        )
        if match:
            json_text = match.group(1)
            # Unescape JSON escapes only
            json_text = json_text.replace(r"\/", "/")
            json_ld_raw = json.loads(json_text)
            json_ld = cast(Dict[str, Any], json_ld_raw)

            # Decode HTML entities in description
            if json_ld.get("description"):
                json_ld["description"] = html_lib.unescape(
                    cast(str, json_ld["description"])
                )

            # Fix malformed image URLs (AnimePlanet bug: double base_url)
            if json_ld.get("image") and "anime-planet.comhttps://" in cast(
                str, json_ld["image"]
            ):
                json_ld["image"] = cast(str, json_ld["image"]).replace(
                    "https://www.anime-planet.comhttps://", "https://"
                )

            return json_ld
    except (json.JSONDecodeError, AttributeError) as e:
        logging.warning(f"Failed to extract JSON-LD: {e}")
    return None


def _extract_rank(rank_texts: List[Dict[str, str]]) -> Optional[int]:
    """Extract popularity rank from entry bar text."""
    for item in rank_texts:
        text = item.get("text", "")
        # Look for "Rank #N" or "#N"
        rank_match = re.search(r"#(\d+)", text)
        if rank_match:
            return int(rank_match.group(1))
    return None


def _extract_studios(studios_raw: List[Dict[str, str]]) -> List[str]:
    """Extract unique studio names."""
    studios = []
    for item in studios_raw:
        studio = item.get("studio", "").strip()
        if studio and studio not in studios:
            studios.append(studio)
    return studios[:5]  # Limit to 5 main studios


def _determine_season_from_date(date_str: str) -> Optional[str]:
    """Determine anime season from start date string."""
    if not date_str:
        return None

    # Extract month from date
    month_match = re.search(r"-(\d{2})-", date_str)
    if not month_match:
        return None

    month = int(month_match.group(1))
    if month in [12, 1, 2]:
        return "WINTER"
    elif month in [3, 4, 5]:
        return "SPRING"
    elif month in [6, 7, 8]:
        return "SUMMER"
    elif month in [9, 10, 11]:
        return "FALL"

    return None


def _process_related_anime(
    related_anime_raw: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Process related anime data from raw extracted list."""
    related_anime = []

    for item in related_anime_raw:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()

        if not title or not url:
            continue

        # Extract slug from URL
        slug_match = re.search(r"/anime/([^/?]+)", url)
        slug = slug_match.group(1) if slug_match else None

        if not slug:
            continue

        related_item = {
            "title": title,
            "slug": slug,
            "url": f"https://www.anime-planet.com{url}",
            "relation_type": "same_franchise",
        }

        # Add relation subtype if present (Sequel, Prequel, etc.)
        relation_subtype = item.get("relation_subtype", "").strip()
        if relation_subtype:
            related_item["relation_subtype"] = relation_subtype.upper()

        # Extract dates from data attributes (preferred method)
        start_date = item.get("start_date_attr", "").strip()
        end_date = item.get("end_date_attr", "").strip()

        if start_date:
            related_item["start_date"] = start_date
            # Extract year from start date
            year_match = re.search(r"(\d{4})", start_date)
            if year_match:
                related_item["year"] = int(year_match.group(1))

        if end_date:
            related_item["end_date"] = end_date
            # Extract year from end date if no start date
            if not start_date:
                year_match = re.search(r"(\d{4})", end_date)
                if year_match:
                    related_item["year"] = int(year_match.group(1))

        # Parse metadata text (contains: type, episodes)
        metadata_text = item.get("metadata_text", "")
        if metadata_text:
            # Extract type and episodes (e.g., "TV: 12 ep")
            type_ep_match = re.search(
                r"(Web|TV Special|TV|OVA|Movie|Special|ONA|Music\s*Video)(?:\s*:\s*(\d+)\s*ep)?",
                metadata_text,
                re.IGNORECASE,
            )
            if type_ep_match:
                related_item["type"] = type_ep_match.group(1).strip().upper()
                if type_ep_match.group(2):
                    related_item["episodes"] = int(type_ep_match.group(2))

        related_anime.append(related_item)

    return related_anime


def _process_related_manga(
    related_manga_raw: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Process related manga data from raw extracted list."""
    related_manga = []

    for item in related_manga_raw:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()

        if not title or not url:
            continue

        # Extract slug from URL
        slug_match = re.search(r"/manga/([^/?]+)", url)
        slug = slug_match.group(1) if slug_match else None

        if not slug:
            continue

        related_item = {
            "title": title,
            "slug": slug,
            "url": f"https://www.anime-planet.com{url}",
            "relation_type": "same_franchise",
        }

        # Add relation subtype if present (Sequel, Prequel, etc.)
        relation_subtype = item.get("relation_subtype", "").strip()
        if relation_subtype:
            related_item["relation_subtype"] = relation_subtype.upper()

        # Extract dates from data attributes (preferred method)
        start_date = item.get("start_date_attr", "").strip()
        end_date = item.get("end_date_attr", "").strip()

        if start_date:
            related_item["start_date"] = start_date
            # Extract year from start date
            year_match = re.search(r"(\d{4})", start_date)
            if year_match:
                related_item["year"] = int(year_match.group(1))

        if end_date:
            related_item["end_date"] = end_date
            # Extract year from end date if no start date
            if not start_date:
                year_match = re.search(r"(\d{4})", end_date)
                if year_match:
                    related_item["year"] = int(year_match.group(1))

        # Parse metadata text (contains: type, volumes, chapters)
        metadata_text = item.get("metadata_text", "")
        if metadata_text:
            # Example: "Vol: 1, Ch: 5"
            vol_match = re.search(r"Vol:\s*([\d?]+)", metadata_text, re.IGNORECASE)
            if vol_match:
                vol_raw = vol_match.group(1)
                if vol_raw.isdigit():
                    related_item["volumes"] = int(vol_raw)

            ch_match = re.search(r"Ch:\s*([\d?]+)", metadata_text, re.IGNORECASE)
            if ch_match:
                ch_raw = ch_match.group(1)
                if ch_raw.isdigit():
                    related_item["chapters"] = int(ch_raw)
        related_manga.append(related_item)

    return related_manga


async def main() -> int:
    """CLI entry point for anime-planet.com crawler."""
    parser = argparse.ArgumentParser(
        description="Crawl anime data from anime-planet.com"
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="Anime identifier: slug (e.g., 'dandadan'), path (e.g., '/anime/dandadan'), or full URL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="animeplanet_anime.json",
        help="Output file path (default: animeplanet_anime.json in current directory)",
    )
    args = parser.parse_args()

    try:
        await fetch_animeplanet_anime(
            args.identifier,
            return_data=False,  # CLI doesn't need return value
            output_path=args.output,
        )
    except Exception:
        logging.exception("Failed to fetch anime-planet anime data")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
