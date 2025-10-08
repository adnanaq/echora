"""
This script crawls anime information from a given anisearch.com anime URL.

It accepts a URL as a command-line argument. It then uses the crawl4ai
library to extract anime data based on a predefined CSS schema.
The extracted data is processed to clean it up.

The final processed data, a dictionary of anime details, is saved
to 'anisearch_anime.json' in the project root.

Usage:
    python anime_crawler.py <anisearch_url>
"""

import argparse
import asyncio
import html  # Import the html module for unescaping HTML entities
import json
import re
from typing import Any, Callable, Dict, List, Optional, cast

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn


def clean_field(text: str) -> str:
    """
    Helper function to clean text fields by removing leading headers like 'Type:'.
    """
    if text:  # Check if text is not None or empty
        # This regex matches any characters from the start of the string up to the first colon, followed by optional whitespace.
        return re.sub(r"^\s*[^:]+:\s*", "", text).strip()
    return text


def _process_relation_tooltips(relations_list: List[Dict[str, Any]]):
    """
    Processes a list of relations to extract image URLs from tooltip_html and renames the field.
    """
    for relation in relations_list:
        if "image" in relation and relation["image"]:
            unescaped_html = html.unescape(relation["image"])
            img_match = re.search(r'<img src="([^"]+)"', unescaped_html)
            if img_match:
                relation["image"] = img_match.group(1)  # Set image to processed URL


async def _fetch_and_process_sub_page(
    crawler: AsyncWebCrawler,
    relative_url: str,
    css_schema: Dict[str, Any],
    url_modifier: Optional[Callable[[str], str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generic function to fetch data from a sub-page. Returns the raw extracted content.
    """
    if not relative_url:
        return None

    if url_modifier:
        relative_url = url_modifier(relative_url)

    # Use the hardcoded base URL
    full_url = f"https://www.anisearch.com/{relative_url.lstrip('/')}"

    extraction_strategy = JsonCssExtractionStrategy(css_schema)
    config = CrawlerRunConfig(extraction_strategy=extraction_strategy)
    results: RunManyReturn = await crawler.arun(full_url, config=config)

    if results:
        for result in results:
            result = cast(CrawlResult, result)
            if result.success and result.extracted_content:
                sub_page_data = json.loads(result.extracted_content)
                if sub_page_data:
                    return sub_page_data[0]
    return None


BASE_ANIME_URL = "https://www.anisearch.com/"


async def fetch_anisearch_anime(url: str = BASE_ANIME_URL):
    """
    Crawls, processes, and saves anime data from a given anisearch.com URL.
    """
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(f"Invalid URL: URL must start with '{BASE_ANIME_URL}'")

    css_schema = {
        "baseSelector": "body",
        "fields": [
            {
                "name": "cover_image",
                "selector": "section#information img#details-cover",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "japanese_title",
                "selector": "section#information div.title[lang='ja'] strong.f16",
                "type": "text",
            },
            {
                "name": "japanese_title_alt",
                "selector": "section#information div.title[lang='ja'] div.grey",
                "type": "text",
            },
            {
                "name": "type",
                "selector": "section#information div.type",
                "type": "text",
            },
            {
                "name": "status",
                "selector": "section#information div.status",
                "type": "text",
            },
            {
                "name": "published",
                "selector": "section#information div.released",
                "type": "text",
            },
            {
                "name": "studio",
                "selector": "section#information div.company a[href*='company']",
                "type": "text",
            },
            {
                "name": "staff",
                "selector": "section#information div.creators a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "source_material",
                "selector": "section#information div.adapted",
                "type": "text",
            },
            {
                "name": "websites",
                "selector": "section#information div.websites a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "english_title",
                "selector": "section#information div.title[lang='en'] strong.f16",
                "type": "text",
            },
            {
                "name": "english_status",
                "selector": "section#information ul.xlist > li:nth-child(2) div.status",
                "type": "text",
            },
            {
                "name": "english_published",
                "selector": "section#information ul.xlist > li:nth-child(2) div.released",
                "type": "text",
            },
            {
                "name": "publishers",
                "selector": "section#information ul.xlist > li:nth-child(2) div.company a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "synonyms",
                "selector": "section#information div.synonyms",
                "type": "text",
            },
            {
                "name": "description",
                "selector": "section#description div.textblock.details-text",
                "type": "text",
            },
            {
                "name": "genres",
                "selector": "section#genres-tags ul.cloud a.gg, section#genres-tags ul.cloud a.gc",
                "type": "list",
                "fields": [{"name": "genre", "type": "text"}],
            },
            {
                "name": "tags",
                "selector": "section#genres-tags ul.cloud a.gt",
                "type": "list",
                "fields": [{"name": "tag", "type": "text"}],
            },
            {
                "name": "screenshots_page_url",
                "selector": "section#images div.showall a.sbuttonA",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "relations_page_url",
                "selector": "section#relations div.showall a.sbuttonA",
                "type": "attribute",
                "attribute": "href",
            },
        ],
    }

    async with AsyncWebCrawler() as crawler:
        extraction_strategy = JsonCssExtractionStrategy(css_schema)
        config = CrawlerRunConfig(extraction_strategy=extraction_strategy)

        results: RunManyReturn = await crawler.arun(url=url, config=config)

        if not results:
            print("No results found.")
            return

        for result in results:
            result = cast(CrawlResult, result)

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if not data:
                    print("Extraction returned empty data.")
                    return

                anime_data = data[0]

                # Clean up the data
                for field in [
                    "type",
                    "status",
                    "published",
                    "source_material",
                    "synonyms",
                    "english_status",
                    "english_published",
                ]:
                    if field in anime_data and isinstance(anime_data[field], str):
                        anime_data[field] = clean_field(anime_data[field])

                # Extract start_date and end_date from published field
                anime_data["start_date"] = None
                anime_data["end_date"] = None
                if "published" in anime_data and anime_data["published"]:
                    published_str = anime_data["published"]
                    match = re.search(
                        r"(\d{2}\.\d{2}\.\d{4})\s*‑\s*(\d{2}\.\d{2}\.\d{4})",
                        published_str,
                    )
                    if match:
                        anime_data["start_date"] = match.group(1)
                        anime_data["end_date"] = match.group(2)
                    else:
                        single_date_match = re.search(
                            r"(\d{2}\.\d{2}\.\d{4})", published_str
                        )
                        if single_date_match:
                            anime_data["start_date"] = single_date_match.group(0)
                            anime_data["end_date"] = None

                if "published" in anime_data:
                    del anime_data["published"]

                # Flatten genres and tags
                for field in ["genres", "tags"]:
                    if field in anime_data and isinstance(anime_data[field], list):
                        anime_data[field] = [
                            item[field[:-1]]
                            for item in anime_data[field]
                            if field[:-1] in item
                        ]

                # Define schemas for sub-pages
                screenshots_css_schema = {
                    "baseSelector": "body",
                    "fields": [
                        {
                            "name": "screenshot_urls",
                            "selector": "div#screenshots a.zoom",
                            "type": "nested_list",
                            "fields": [
                                {
                                    "name": "url",
                                    "type": "attribute",
                                    "attribute": "href",
                                }
                            ],
                        }
                    ],
                }

                relation_fields = [
                    {"name": "type", "selector": "th span", "type": "text"},
                    {"name": "title", "selector": "th a", "type": "text"},
                    {
                        "name": "url",
                        "selector": "th a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "details",
                        "selector": "td[data-title='Type / Episodes / Year']",
                        "type": "text",
                    },
                    {
                        "name": "genres",
                        "selector": "td[data-title='Main genres']",
                        "type": "text",
                    },
                    {
                        "name": "rating",
                        "selector": "td.rating div.star0",
                        "type": "attribute",
                        "attribute": "title",
                    },
                    {
                        "name": "image",
                        "selector": "th[scope='row']",
                        "type": "attribute",
                        "attribute": "data-tooltip",
                    },
                ]

                relations_css_schema = {
                    "baseSelector": "body",
                    "fields": [
                        {
                            "name": "anime_relations",
                            "selector": "section#relations_anime tbody tr",
                            "type": "nested_list",
                            "fields": relation_fields,
                        },
                        {
                            "name": "manga_relations",
                            "selector": "section#relations_manga tbody tr",
                            "type": "nested_list",
                            "fields": relation_fields,
                        },
                    ],
                }

                # Crawl screenshots
                screenshots_raw_data = await _fetch_and_process_sub_page(
                    crawler,
                    anime_data.get("screenshots_page_url", ""),
                    screenshots_css_schema,
                )
                if screenshots_raw_data and "screenshot_urls" in screenshots_raw_data:
                    anime_data["screenshots"] = [
                        item["url"]
                        for item in screenshots_raw_data["screenshot_urls"]
                        if "url" in item
                    ]
                else:
                    anime_data["screenshots"] = []
                if "screenshots_page_url" in anime_data:
                    del anime_data["screenshots_page_url"]

                # Crawl relations
                relations_raw_data = await _fetch_and_process_sub_page(
                    crawler,
                    anime_data.get("relations_page_url", ""),
                    relations_css_schema,
                    url_modifier=lambda u: (
                        u + "?show=overall" if "?show=overall" not in u else u
                    ),
                )
                if relations_raw_data:
                    if "anime_relations" in relations_raw_data:
                        _process_relation_tooltips(
                            relations_raw_data["anime_relations"]
                        )
                        anime_data["anime_relations"] = relations_raw_data[
                            "anime_relations"
                        ]
                    else:
                        anime_data["anime_relations"] = []

                    if "manga_relations" in relations_raw_data:
                        _process_relation_tooltips(
                            relations_raw_data["manga_relations"]
                        )
                        anime_data["manga_relations"] = relations_raw_data[
                            "manga_relations"
                        ]
                    else:
                        anime_data["manga_relations"] = []
                else:
                    anime_data["anime_relations"] = []
                    anime_data["manga_relations"] = []

                if "relations_page_url" in anime_data:
                    del anime_data["relations_page_url"]

                output_path = (
                    "/home/dani/code/anime-vector-service/anisearch_anime.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(anime_data, f, ensure_ascii=False, indent=2)
                print(f"Data written to {output_path}")
            else:
                print(f"Extraction failed: {result.error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl anime data from an anisearch.com URL."
    )
    parser.add_argument(
        "url", type=str, help="The anisearch.com URL for the anime page."
    )
    args = parser.parse_args()
    asyncio.run(fetch_anisearch_anime(args.url))

