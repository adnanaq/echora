"""
Crawls anime information from anime-planet.com anime pages with Redis caching.

Extracts comprehensive anime data including related anime, rankings, studios,
and all metadata from JSON-LD. Results are cached in Redis for 24 hours to avoid
repeated crawling.

Usage:
    ./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_anime_crawler.py -- <identifier> [--output PATH]

    <identifier>    anime-planet.com anime identifier (slug, path, or full URL)
    --output PATH   optional output file path (default: animeplanet_anime.json)
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable
from typing import Any, cast

from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_status,
    determine_anime_year,
)
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn
from enrichment.crawlers.utils import sanitize_output_path
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

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
    """
    Extract the anime slug from a canonical Anime-Planet anime URL.

    Parameters:
        url (str): An Anime-Planet anime URL or path containing "/anime/<slug>".

    Returns:
        str: The slug segment following "/anime/".

    Raises:
        ValueError: If a slug cannot be found in the provided URL.
    """
    # Extract slug from: https://www.anime-planet.com/anime/dandadan
    match = re.search(r"/anime/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Could not extract slug from URL: {url}")
    return match.group(1)


@cached_result(ttl=TTL_ANIME_PLANET, key_prefix="animeplanet_anime")
async def _fetch_animeplanet_anime_data(
    canonical_slug: str,
) -> dict[str, Any] | None:
    """
    Fetches and assembles comprehensive anime metadata for a given canonical anime-planet slug.

    Data is extracted from the anime's page (JSON-LD and page sections); the function performs no side effects and is cached by canonical slug.

    Parameters:
        canonical_slug (str): Canonical anime slug normalized by the caller (e.g., "dandadan").

    Returns:
        Complete anime data dictionary with consolidated fields (titles, dates, genres, studios, related items, poster, rank, status, etc.), or `None` if extraction fails.
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

        logger.info(f"Fetching anime data: {url}")
        results: RunManyReturn = await crawler.arun(url=url, config=config)

        if not results:
            logger.warning("No results found.")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if not data:
                    logger.warning("Extraction returned empty data.")
                    return None

                anime_data = cast(dict[str, Any], data[0])

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

                # Add slug (passed as canonical_slug parameter)
                anime_data["slug"] = canonical_slug

                # Process statistics
                rank = _extract_rank(
                    cast(list[dict[str, str]], anime_data.get("rank_text", []))
                )

                score = None
                scored_by = 0
                if json_ld and json_ld.get("aggregateRating"):
                    ar = json_ld["aggregateRating"]
                    score = ar.get("ratingValue")
                    scored_by = ar.get("ratingCount", 0)

                anime_data["statistics"] = {
                    "score": score,
                    "scored_by": scored_by,
                    "rank": rank,
                    "popularity": None,  # Anime-Planet doesn't distinguish between rank and popularity
                    "favorites": None,
                }

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
                start_date = None
                end_date = None

                if json_ld and json_ld.get("startDate"):
                    start_date = json_ld["startDate"]
                    end_date = json_ld.get("endDate")

                    # Extract year using utility function
                    year = determine_anime_year(start_date)
                    if year:
                        anime_data["year"] = year

                    # Determine season using utility function
                    season = determine_anime_season(start_date)
                    if season:
                        anime_data["season"] = season

                # Determine status using utility function
                anime_data["status"] = determine_anime_status(start_date, end_date)

                # Return pure data (no side effects)
                return anime_data
            else:
                logger.warning(f"Extraction failed: {result.error_message}")
                return None


async def fetch_animeplanet_anime(
    slug: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """
    Fetch anime data for an Anime-Planet identifier, optionally write it to disk, and return it.

    Normalize the provided slug/URL, obtain the canonical anime data (may be returned from cache), write the JSON to `output_path` if provided, and return the data.

    Parameters:
        slug (str): Anime identifier — a slug (e.g., "dandadan"), a path (e.g., "/anime/dandadan"), or a full URL.
        output_path (Optional[str]): If provided, write the resulting JSON to this file path.

    Returns:
        dict: Complete anime data dictionary if data was found, `None` otherwise.
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
        logger.info(f"Data written to {safe_path}")

    return data


def _extract_json_ld(html: str) -> dict[str, Any] | None:
    """
    Extract JSON-LD structured data from an HTML document.

    Parses the first <script type="application/ld+json"> block in the provided HTML and returns its content as a dictionary. If present, HTML entities in the `description` field are unescaped and known malformed image URLs are corrected.

    Parameters:
        html (str): Full HTML source of a page.

    Returns:
        Dict[str, Any]: Parsed JSON-LD object when extraction succeeds, `None` otherwise.
    """
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
            json_ld = cast(dict[str, Any], json_ld_raw)

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
        logger.warning(f"Failed to extract JSON-LD: {e}")
    return None


def _extract_rank(rank_texts: list[dict[str, str]]) -> int | None:
    """
    Determine the numeric popularity rank from a list of entry-bar text items.

    Parameters:
        rank_texts (List[Dict[str, str]]): List of extracted text items (each dict should include a "text" key) to search for a rank.

    Returns:
        int or None: The first integer rank found (for example, 123), or `None` if no rank is present.
    """
    for item in rank_texts:
        text = item.get("text", "")
        # Look for "Rank #N" or "#N"
        rank_match = re.search(r"#(\d+)", text)
        if rank_match:
            return int(rank_match.group(1))
    return None


def _extract_studios(studios_raw: list[dict[str, str]]) -> list[str]:
    """Extract unique studio names."""
    studios = []
    for item in studios_raw:
        studio = item.get("studio", "").strip()
        if studio and studio not in studios:
            studios.append(studio)
    return studios[:5]  # Limit to 5 main studios


def _parse_anime_metadata(metadata_text: str, related_item: dict[str, Any]) -> None:
    """
    Parse anime-specific metadata and populate the related item with `type` and `episodes` when present.

    Parameters:
        metadata_text (str): Metadata string containing type and optional episode count (e.g., "TV: 12 ep").
        related_item (Dict[str, Any]): Mutable mapping representing the related anime; this function sets `type` (uppercased) and `episodes` (int) in place.
    """
    if not metadata_text:
        return

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


def _parse_manga_metadata(metadata_text: str, related_item: dict[str, Any]) -> None:
    """
    Parse manga metadata text and populate numeric `volumes` and `chapters` fields on the provided related item.

    Extracts "Vol: N" and "Ch: N" patterns from the metadata text and sets related_item["volumes"] and related_item["chapters"] to the parsed integers when numeric values are present. Non-numeric or missing values are ignored. The function mutates the given related_item in place.

    Parameters:
        metadata_text (str): Raw metadata string (e.g., "Vol: 1, Ch: 5").
        related_item (Dict[str, Any]): Dictionary representing a related manga item to be updated.
    """
    if not metadata_text:
        return

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


def _process_related_items(
    items_raw: list[dict[str, Any]],
    item_type: str,
    metadata_parser: Callable[[str, dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """
    Normalize and enrich raw related anime/manga entries into structured related-item records.

    Parameters:
        items_raw (List[Dict[str, Any]]): Raw extracted items containing keys like `title`, `url`, `relation_subtype`, `start_date_attr`, `end_date_attr`, and `metadata_text`.
        item_type (str): Either `"anime"` or `"manga"`, used to locate and extract the slug from the item's URL path.
        metadata_parser (Callable[[str, Dict[str, Any]], None]): Function that parses the item's `metadata_text` and mutates the provided related-item dict with type-specific fields (for example, `episodes`, `volumes`, `chapters`, or `type`).

    Returns:
        List[Dict[str, Any]]: A list of processed related-item dictionaries. Each dictionary always includes `title`, `slug`, `url` (absolute, prefixed with "https://www.anime-planet.com"), and `relation_type`. Optional fields produced when present include `relation_subtype` (uppercased), `start_date`, `end_date`, `year` (integer extracted from dates), and any additional fields populated by `metadata_parser`.
    """
    processed_items = []
    url_pattern = rf"/{item_type}/([^/?]+)"

    for item in items_raw:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()

        if not title or not url:
            continue

        # Extract slug from URL
        slug_match = re.search(url_pattern, url)
        slug = slug_match.group(1) if slug_match else None

        if not slug:
            continue

        # Build absolute URL, handling both relative and absolute href values
        full_url = url
        if not full_url.startswith("http"):
            full_url = f"https://www.anime-planet.com{full_url}"

        related_item: dict[str, Any] = {
            "title": title,
            "slug": slug,
            "url": full_url,
            "relation_type": "same_franchise",
        }

        # Add relation subtype if present (Sequel, Prequel, etc.)
        relation_subtype = item.get("relation_subtype", "").strip()
        if relation_subtype:
            related_item["relation_subtype"] = relation_subtype.upper()

        # Preserve image URL when available
        image = item.get("image", "").strip()
        if image:
            related_item["image"] = image

        # Extract dates from data attributes (preferred method)
        start_date = item.get("start_date_attr", "").strip()
        end_date = item.get("end_date_attr", "").strip()

        if start_date:
            related_item["start_date"] = start_date
            # Extract year from start date using utility function
            year = determine_anime_year(start_date)
            if year:
                related_item["year"] = year

        if end_date:
            related_item["end_date"] = end_date
            # Extract year from end date if no start date using utility function
            if not start_date:
                year = determine_anime_year(end_date)
                if year:
                    related_item["year"] = year

        # Parse type-specific metadata
        metadata_text = item.get("metadata_text", "")
        metadata_parser(metadata_text, related_item)

        processed_items.append(related_item)

    return processed_items


def _process_related_anime(
    related_anime_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize and enrich raw related-anime entries extracted from an Anime-Planet page.

    Process each raw item to produce a canonical related-anime record: extract and normalize the slug and absolute URL, uppercase and attach relation subtype when present, parse start/end dates and derive a year when available, and parse metadata to populate `type` (uppercased) and numeric `episodes` when present.

    Parameters:
        related_anime_raw (List[Dict[str, Any]]): Raw extracted items (each typically contains keys such as `title`, `url`, `relation_subtype`, `image`, `start_date_attr`, `end_date_attr`, and `metadata_text`).

    Returns:
        List[Dict[str, Any]]: A list of processed related-anime dictionaries. Typical keys include:
            - title (str)
            - slug (str)
            - url (str) — absolute Anime-Planet URL
            - relation_type (str) — set to "same_franchise"
            - relation_subtype (str) — uppercased, if present
            - start_date (str) / end_date (str) — if present
            - year (int) — derived from dates when available
            - type (str) — uppercased media type when parsed from metadata
            - episodes (int) — when parsed from metadata
            - image (str) and other preserved fields from the raw item
    """
    return _process_related_items(related_anime_raw, "anime", _parse_anime_metadata)


def _process_related_manga(
    related_manga_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert a list of raw related-manga entries into normalized related-manga records.

    Parameters:
        related_manga_raw (List[Dict[str, Any]]): Raw extraction items for related manga. Each item is expected to include at least `title` and `url` and may contain `relation_subtype`, `start_date_attr`, `end_date_attr`, and `metadata_text` (e.g., "Vol: X Ch: Y").

    Returns:
        List[Dict[str, Any]]: Normalized related-manga objects containing:
            - title (str)
            - slug (str): extracted slug from the manga URL
            - url (str): absolute anime-planet URL
            - relation_type (str): set to "same_franchise"
            - relation_subtype (str, optional): uppercased subtype if present
            - start_date (str, optional)
            - end_date (str, optional)
            - year (int, optional): extracted from start_date or end_date
            - volumes (int, optional)
            - chapters (int, optional)
    """
    return _process_related_items(related_manga_raw, "manga", _parse_manga_metadata)


async def main() -> int:
    """
    Run the CLI to crawl a single anime page from anime-planet.com and optionally write the result to a JSON file.

    Parses command-line arguments (anime identifier and optional --output path), invokes the crawler, and exits with a process-style code.

    Returns:
        exit_code (int): 0 on success, 1 on failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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
        data = await fetch_animeplanet_anime(
            args.identifier,
            output_path=args.output,
        )
        if data is None:
            logger.error("No data was extracted; see logs above for details.")
            return 1
    except (ValueError, OSError):
        logger.exception("Failed to fetch anime-planet anime data")
        return 1
    except Exception:
        logger.exception("Unexpected error during anime fetch")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
