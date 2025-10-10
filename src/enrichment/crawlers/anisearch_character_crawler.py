"""
This script crawls character information from a given anisearch.com anime URL.

It accepts a URL as a command-line argument. It then uses the crawl4ai
library to extract character data based on a predefined CSS schema.
The extracted data is processed to clean up fields like 'favorites' and 'image',
and the character's role is added.

The final processed data, a list of characters with their details, is saved
to 'anisearch_characters.json' in the project root.

Usage:
    python character_crawler.py <anisearch_url>
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


async def fetch_anisearch_characters(url: str) -> None:
    """
    Crawls, processes, and saves character data from a given anisearch.com URL.

    This function defines a schema for extracting character information,
    including their name, role, URL, favorites count, and image. It then
    initializes a crawler, runs it on the provided anime character page URL,
    processes the returned data to clean and structure it, and finally
    writes the output to a JSON file.

    Args:
        url (str): The URL of the anisearch.com character page to crawl.
    """
    # Define a correct schema for character extraction
    css_schema = {
        "baseSelector": "#chara1, #chara2, #chara3, #chara5, #chara6, #chara7",
        "fields": [
            {"name": "role", "selector": "h2", "type": "text"},  # h2 inside section
            {
                "name": "characters",
                "selector": "li",  # each character inside ul
                "type": "nested_list",
                "fields": [
                    {"name": "name", "selector": "span.title", "type": "text"},
                    {
                        "name": "url",
                        "selector": "a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {"name": "favorites", "selector": "span.favorites", "type": "text"},
                    {
                        "name": "image",
                        "selector": "a",
                        "type": "attribute",
                        "attribute": "style",
                    },
                ],
            },
        ],
    }

    print(f"Using Schema: {json.dumps(css_schema, indent=2)}")

    extraction_strategy = JsonCssExtractionStrategy(css_schema)
    config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        wait_until="networkidle",
        wait_for_images=True,
        scan_full_page=True,
        adjust_viewport_to_content=True,
        delay_before_return_html=0.5,
    )

    async with AsyncWebCrawler() as crawler:
        results: RunManyReturn = await crawler.arun(url, config=config)

        if not results:
            print("No results found.")
            return

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            print(f"URL: {result.url}")
            print(f"Success: {result.success}")

            if result.success and result.extracted_content:
                # The result is a JSON string, so we need to load it first
                data = json.loads(result.extracted_content)
                # Iterate through sections
                flattened_characters = []

                for section in data:
                    role = section.get("role", "").replace("Character", "").strip()
                    for character in section.get("characters", []):
                        # Clean up favorites
                        if "favorites" in character:
                            match = re.search(r"\d+", str(character["favorites"]))
                            if match:
                                character["favorites"] = int(match.group(0))
                            else:
                                del character["favorites"]

                        # Extract image URL from style
                        if "image" in character and character["image"]:
                            match = re.search(
                                r'url\(["\\]?(.*?)["\\]?\)',
                                character["image"],  # Corrected escaping here
                            )
                            if match:
                                character["image"] = match.group(1)

                        # Add role directly into the character object
                        character["role"] = role
                        character["url"] = (
                            "https://www.anisearch.com/" + character["url"]
                        )
                        flattened_characters.append(character)

                output_path = (
                    "/home/dani/code/anime-vector-service/anisearch_characters.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"characters": flattened_characters},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            else:
                print(f"Extraction failed: {result.error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl character data from an anisearch.com URL."
    )
    parser.add_argument(
        "url", type=str, help="The anisearch.com URL for the anime characters page."
    )
    args = parser.parse_args()

    asyncio.run(fetch_anisearch_characters(args.url))
