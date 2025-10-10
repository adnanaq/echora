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
from typing import cast

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn


async def fetch_anisearch_episodes(url: str) -> None:
    """
    Crawls, processes, and saves episode data from a given anisearch.com URL.

    This function defines a schema for extracting episode information,
    including episode number, runtime, release date, and title. It then
    initializes a crawler, runs it on the provided anime episode page URL,
    processes the returned data to clean it, and finally writes the output
    to a JSON file.

    Args:
        url (str): The URL of the anisearch.com episode page to crawl.
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
            return

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

                output_path = (
                    "/home/dani/code/anime-vector-service/anisearch_episodes.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Data written to {output_path}")
            else:
                print(f"Extraction failed: {result.error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl episode data from an anisearch.com URL."
    )
    parser.add_argument(
        "url", type=str, help="The anisearch.com URL for the anime episodes page."
    )
    args = parser.parse_args()
    asyncio.run(fetch_anisearch_episodes(args.url))
