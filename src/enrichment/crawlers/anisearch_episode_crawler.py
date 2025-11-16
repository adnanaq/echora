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

from src.cache_manager.config import get_cache_config
from src.cache_manager.result_cache import cached_result

from .utils import sanitize_output_path

# Get TTL from config to keep cache control centralized
_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

BASE_EPISODE_URL = "https://www.anisearch.com/anime/"


def _normalize_episode_url(anime_identifier: str) -> str:
    """
    Normalize various input formats to full anisearch episode URL.

    Accepts:
        - Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"
        - Path: "/18878,dan-da-dan/episodes" or "18878,dan-da-dan/episodes"
        - Anime ID: "18878,dan-da-dan" (will append /episodes)

    Returns:
        Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"
    """
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
    Extract anime ID from episode URL.

    Args:
        url: Episode URL

    Returns:
        Anime ID (e.g., "18878,dan-da-dan")
    """
    # Remove base URL and /episodes suffix
    if url.startswith(BASE_EPISODE_URL):
        path = url[len(BASE_EPISODE_URL):]
        # Remove /episodes suffix
        return path.replace("/episodes", "").rstrip("/")

    raise ValueError(f"Invalid episode URL: {url}")


@cached_result(ttl=TTL_ANISEARCH, key_prefix="anisearch_episodes")
async def _fetch_anisearch_episodes_data(canonical_anime_id: str) -> Optional[list[dict[str, Any]]]:
    """
    Pure cached function that fetches episode data from anisearch.com.

    Cache key depends ONLY on canonical anime ID, not on output_path or return_data.
    This ensures efficient cache usage across different output destinations.

    Args:
        canonical_anime_id: Canonical anime ID (e.g., "18878,dan-da-dan") - already normalized by caller

    Returns:
        List of episode dictionaries, or None if fetch fails.
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
            logging.warning("No results found.")
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
            else:
                logging.warning(f"Extraction failed: {result.error_message}")
                return None


async def fetch_anisearch_episodes(
    anime_id: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[list[dict[str, Any]]]:
    """
    Wrapper function that handles side effects (file writing, return_data logic).

    This function calls the cached _fetch_anisearch_episodes_data() to get the data,
    then performs side effects that should execute regardless of cache status.

    Args:
        anime_id: Anime identifier - can be:
            - Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"
            - Path: "/18878,dan-da-dan/episodes" or "18878,dan-da-dan/episodes"
            - Anime ID: "18878,dan-da-dan" (will auto-append /episodes)
        return_data: Whether to return the data (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        List of episode dictionaries (if return_data=True), otherwise None
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
        logging.info(f"Data written to {safe_path}")

    # Return based on return_data parameter
    if return_data:
        return data

    return None


async def main() -> int:
    """CLI entry point for anisearch.com episode crawler."""
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
    except Exception:
        logging.exception("Failed to fetch anisearch episode data")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
