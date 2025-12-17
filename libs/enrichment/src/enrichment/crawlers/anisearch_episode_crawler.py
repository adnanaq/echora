"""
Crawls episode information from anisearch.com anime URLs with Redis caching.

Extracts episode data using crawl4ai with CSS selectors and converts episode
numbers to integers. Results are cached in Redis for 24 hours to avoid
repeated crawling.

Usage:
    python -m src.enrichment.crawlers.anisearch_episode_crawler <url> [--output PATH]

    <url>           anisearch.com anime episode page URL
    --output PATH   optional output file path (default: anisearch_episodes.json)
"""

import argparse
import asyncio
import json
import logging

import re
import sys
from typing import Any, Optional, cast

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

from .utils import sanitize_output_path

logger = logging.getLogger(__name__)

# Get TTL from config to keep cache control centralized
_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

BASE_EPISODE_URL = "https://www.anisearch.com/anime/"

# Error message constants (for Ruff TRY003 compliance)
_INVALID_EPISODE_URL_MSG = "Invalid episode URL: must start with {base}"


def _normalize_episode_url(anime_identifier: str) -> str:
    """
    Normalize an anime identifier into a full Anisearch episodes page URL.

    Accepts a full episodes URL, a full anime URL without the `/episodes` suffix, a path (with or without leading `/`), or a canonical anime id (e.g. `18878,dan-da-dan`) and returns the canonical episodes URL.

    Parameters:
        anime_identifier (str): Full URL, path, or canonical anime id identifying the anime.

    Returns:
        str: Full Anisearch episodes page URL (e.g. "https://www.anisearch.com/anime/18878,dan-da-dan/episodes").

    Raises:
        ValueError: If anime_identifier is an HTTP(S) URL but not an anisearch.com domain.
    """
    # Reject non-anisearch HTTP(S) URLs early for transparency
    if anime_identifier.startswith(("http://", "https://")) and "anisearch.com" not in anime_identifier:
        msg = f"Invalid URL: expected anisearch.com domain, got {anime_identifier}"
        raise ValueError(msg)

    # If already full URL with /episodes, return as-is
    if anime_identifier.startswith("https://www.anisearch.com/anime/") and "/episodes" in anime_identifier:
        return anime_identifier

    # If full URL without /episodes, append it
    if anime_identifier.startswith("https://www.anisearch.com/anime/"):
        return f"{anime_identifier.rstrip('/')}/episodes"

    # Remove leading slash if present
    clean_id = anime_identifier.lstrip("/")

    # Remove /episodes suffix if present (we'll add it back)
    clean_id = clean_id.replace("/episodes", "").rstrip("/")

    # Construct full URL
    return f"{BASE_EPISODE_URL}{clean_id}/episodes"


def _extract_anime_id_from_episode_url(url: str) -> str:
    """
    Extract the canonical anime identifier from an Anisearch episode page URL.
    
    Parameters:
        url (str): Full episode page URL that begins with the module's BASE_EPISODE_URL and may end with `/episodes`.
    
    Returns:
        str: Canonical anime ID extracted from the URL (for example, "18878,dan-da-dan").
    
    Raises:
        ValueError: If `url` does not start with the expected base episode URL.
    """
    # Remove base URL and /episodes suffix
    if url.startswith(BASE_EPISODE_URL):
        path = url[len(BASE_EPISODE_URL):]
        # Remove /episodes suffix
        return path.replace("/episodes", "").rstrip("/")

    raise ValueError(_INVALID_EPISODE_URL_MSG.format(base=BASE_EPISODE_URL))


@cached_result(ttl=TTL_ANISEARCH, key_prefix="anisearch_episodes")
async def _fetch_anisearch_episodes_data(canonical_anime_id: str) -> Optional[list[dict[str, Any]]]:
    """
    Fetches episode metadata for an anime from anisearch.com using its canonical anime ID.
    
    Results are cached by canonical anime ID (decorator-managed); callers should treat this function as a pure, cache-keyed data fetcher.
    
    Parameters:
        canonical_anime_id (str): Canonical anime identifier portion used in anisearch episode URLs (e.g., "18878,dan-da-dan").
    
    Returns:
        list[dict[str, Any]] | None: A list of episode dictionaries when extraction succeeds, or `None` if no results or extraction fails.
            Each episode dictionary contains keys: `episodeNumber` (int when present), `runtime` (str), `releaseDate` (str), and `title` (str).
    """
    # Build URL from canonical anime ID (caller already normalized)
    url = f"{BASE_EPISODE_URL}{canonical_anime_id}/episodes"
    css_schema = {
        "baseSelector": "tr[data-episode='true']",
        "fields": [
            {
                "name": "episodeNumber",
                "selector": "th[itemprop='episodeNumber']",
                "type": "text",
            },
            {
                "name": "runtime",
                "selector": "td[data-title='Runtime'] div[lang='en']",
                "type": "text",
            },
            {
                "name": "releaseDate",
                "selector": "td[data-title='Date of Original Release'] div[lang='en']",
                "type": "text",
            },
            {
                "name": "title",
                "selector": "td[data-title='Title'] span[itemprop='name'][lang='en']",
                "type": "text",
            },
        ],
    }

    extraction_strategy = JsonCssExtractionStrategy(css_schema)
    config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

    async with AsyncWebCrawler() as crawler:
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
                data = cast(list[dict[str, Any]], json.loads(result.extracted_content))
                # Clean up the data
                for item in data:
                    episode_number = item.get("episodeNumber")
                    if episode_number:
                        match = re.search(r"\d+", episode_number)
                        if match:
                            item["episodeNumber"] = int(match.group(0))

                return data

            # Log extraction failure but continue to fallback return
            logger.warning(f"Extraction failed: {result.error_message}")

        # If loop completes without returning, no valid results were found
        return None


async def fetch_anisearch_episodes(
    anime_id: str, output_path: Optional[str] = None
) -> Optional[list[dict[str, Any]]]:
    """
    Fetch episode data for an anime from Anisearch and optionally persist it to a JSON file.

    Parameters:
        anime_id (str): Anime identifier in any of these forms:
            - Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"
            - Path: "/18878,dan-da-dan/episodes" or "18878,dan-da-dan/episodes"
            - Canonical ID: "18878,dan-da-dan" ("/episodes" will be appended)
        output_path (Optional[str]): If provided, write the episode list to this file as JSON.

    Returns:
        list[dict[str, Any]] if data was found, `None` otherwise.
    """
    # Normalize identifier once so cache keys depend on canonical anime ID
    # This ensures cache reuse across different identifier formats
    episode_url = _normalize_episode_url(anime_id)
    canonical_anime_id = _extract_anime_id_from_episode_url(episode_url)

    # Fetch data from cache or crawl (pure function keyed only on canonical anime ID)
    data = await _fetch_anisearch_episodes_data(canonical_anime_id)

    if data is None:
        return None

    # Side effect: Write to file (always executes, even on cache hit)
    if output_path:
        safe_path = sanitize_output_path(output_path)
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data written to {safe_path}")

    return data


async def main() -> int:
    """
    Run the CLI to fetch episode data from anisearch.com and optionally write it to a JSON file.
    
    The command accepts a positional `anime_id` (full URL, path like '/18878,dan-da-dan/episodes', or canonical ID like '18878,dan-da-dan') and an optional `--output` path (defaults to 'anisearch_episodes.json'). The function invokes the episode fetcher and exits with a status code indicating success or failure.
    
    Returns:
        int: `0` on successful completion, `1` on error.
    """
    parser = argparse.ArgumentParser(
        description="Crawl episode data from anisearch.com."
    )
    parser.add_argument(
        "anime_id",
        type=str,
        help="Anime identifier: full URL, path (e.g., '/18878,dan-da-dan/episodes'), or ID (e.g., '18878,dan-da-dan')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anisearch_episodes.json",
        help="Output file path (default: anisearch_episodes.json in current directory)",
    )
    args = parser.parse_args()

    try:
        await fetch_anisearch_episodes(
            args.anime_id,
            output_path=args.output,
        )
    except (ValueError, OSError):
        logger.exception("Failed to fetch anisearch episode data")
        return 1
    except Exception:
        logger.exception("Unexpected error during episode fetch")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
