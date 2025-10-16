"""
This script crawls episode information from a given anisearch.com anime URL.

It accepts a URL as a command-line argument. It then uses the crawl4ai
library to extract episode data based on a predefined CSS schema.
The extracted data is processed to clean up the episode number.

The final processed data, a list of episodes with their details, is saved
to 'anisearch_episodes.json' in the project root.

Usage:
    python episode_crawler.py <anisearch_url>
"""

import argparse
import asyncio
import json
import re
from typing import Optional, cast

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from .utils import sanitize_output_path


async def fetch_anisearch_episodes(
    url: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[list]:
    """
    Crawls, processes, and saves episode data from a given anisearch.com URL.

    This function defines a schema for extracting episode information,
    including episode number, runtime, release date, and title. It then
    initializes a crawler, runs it on the provided anime episode page URL,
    processes the returned data to clean it, and optionally writes the output
    to a JSON file.

    Args:
        url (str): The URL of the anisearch.com episode page to crawl.
        return_data: Whether to return the data (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        List of episode dictionaries (if return_data=True), otherwise None
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
                data = json.loads(result.extracted_content)
                # Clean up the data
                for item in data:
                    if "episodeNumber" in item and item["episodeNumber"]:
                        match = re.search(r"\d+", item["episodeNumber"])
                        if match:
                            item["episodeNumber"] = int(match.group(0))

                # Conditionally write to file
                if output_path:
                    safe_path = sanitize_output_path(output_path)
                    with open(safe_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"Data written to {safe_path}")

                # Return data for programmatic usage
                if return_data:
                    return data

                return None
            else:
                print(f"Extraction failed: {result.error_message}")
                return None
        return None


if __name__ == "__main__":
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
    asyncio.run(
        fetch_anisearch_episodes(
            args.url,
            return_data=False,  # CLI doesn't need return value
            output_path=args.output,
        )
    )
