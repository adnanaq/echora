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

from src.cache_manager.result_cache import cached_result

from .utils import sanitize_output_path


@cached_result(ttl=86400, key_prefix="anisearch_episodes")  # 24 hours cache
async def _fetch_anisearch_episodes_data(url: str) -> Optional[list[dict[str, Any]]]:
    """
    Pure cached function that fetches episode data from anisearch.com.

    Cache key depends ONLY on the URL parameter, not on output_path or return_data.
    This ensures efficient cache usage across different output destinations.

    Args:
        url (str): The URL of the anisearch.com episode page to crawl.

    Returns:
        List of episode dictionaries, or None if fetch fails.
    """
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
            print("No results found.")
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
                    if "episodeNumber" in item and item["episodeNumber"]:
                        match = re.search(r"\d+", item["episodeNumber"])
                        if match:
                            item["episodeNumber"] = int(match.group(0))

                return data
            else:
                print(f"Extraction failed: {result.error_message}")
                return None
        return None


async def fetch_anisearch_episodes(
    url: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[list[dict[str, Any]]]:
    """
    Wrapper function that handles side effects (file writing, return_data logic).

    This function calls the cached _fetch_anisearch_episodes_data() to get the data,
    then performs side effects that should execute regardless of cache status.

    Args:
        url: The URL of the anisearch.com episode page to crawl
        return_data: Whether to return the data (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        List of episode dictionaries (if return_data=True), otherwise None
    """
    # Fetch data from cache or crawl (pure function)
    data = await _fetch_anisearch_episodes_data(url)

    if data is None:
        return None

    # Side effect: Write to file (always executes, even on cache hit)
    if output_path:
        safe_path = sanitize_output_path(output_path)
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data written to {safe_path}")

    # Return based on return_data parameter
    if return_data:
        return data

    return None


async def main() -> int:
    """CLI entry point for anisearch.com episode crawler."""
    parser = argparse.ArgumentParser(
        description="Crawl episode data from an anisearch.com URL."
    )
    parser.add_argument(
        "url", type=str, help="The anisearch.com URL for the anime episodes page."
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
            args.url,
            return_data=False,  # CLI doesn't need return value
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
