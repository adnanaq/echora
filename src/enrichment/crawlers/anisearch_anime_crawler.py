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
from typing import Any, Dict, List, Optional, cast

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn


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
    url: str,
    session_id: str,
    js_code: str,
    wait_for: str,
    css_schema: Optional[Dict[str, Any]],
    use_js_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Generic function to navigate via JS and fetch data from a sub-page.

    Args:
        crawler: AsyncWebCrawler instance
        url: Base URL (required even for js_only mode)
        session_id: Browser session ID to maintain state
        js_code: JavaScript code to execute for navigation/interaction
        wait_for: CSS selector or JS condition to wait for
        css_schema: CSS extraction schema for data extraction (None for navigation-only)
        use_js_only: If True, continues existing session without full reload

    Returns:
        Extracted data dictionary or None if extraction fails
    """
    extraction_strategy = JsonCssExtractionStrategy(css_schema) if css_schema else None
    config = CrawlerRunConfig(
        session_id=session_id,
        js_code=js_code,
        wait_for=wait_for,
        extraction_strategy=extraction_strategy,
        js_only=use_js_only,
    )

    results: RunManyReturn = await crawler.arun(url=url, config=config)

    if not results:
        return None

    for result in results:
        if not isinstance(result, CrawlResult):
            raise TypeError(
                f"Unexpected result type: {type(result)}, expected CrawlResult."
            )

        if result.success and result.extracted_content:
            sub_page_data = json.loads(result.extracted_content)
            if sub_page_data:
                return sub_page_data[0]


BASE_ANIME_URL = "https://www.anisearch.com/anime/"


async def fetch_anisearch_anime(url: str):
    """
    Crawls, processes, and saves anime data from a given anisearch.com URL.
    Uses JS-based navigation for screenshots and relations pages.

    Args:
        url: Can be either:
            - Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan"
            - Relative path: "/18878,dan-da-dan" or "18878,dan-da-dan"

    Raises:
        ValueError: If URL is not from anisearch.com/anime
    """
    # Normalize the URL
    if not url.startswith("http"):
        # Remove leading slash if present, then construct full URL
        url = f"{BASE_ANIME_URL}{url.lstrip('/')}"

    # Validate it's an anisearch.com anime URL
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(
            f"Invalid URL: Must be an anisearch.com anime URL. "
            f"Expected format: '{BASE_ANIME_URL}<anime-id>' or '/<anime-id>' or '<anime-id>'"
        )

    # Generate unique session ID for maintaining browser state
    import uuid

    session_id = f"anime_session_{uuid.uuid4().hex[:8]}"

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
        ],
    }

    async with AsyncWebCrawler() as crawler:
        extraction_strategy = JsonCssExtractionStrategy(css_schema)
        config = CrawlerRunConfig(
            session_id=session_id, extraction_strategy=extraction_strategy
        )

        print(f"Fetching main page: {url}")
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
                ]:
                    if field in anime_data and isinstance(anime_data[field], str):
                        anime_data[field] = re.sub(
                            r"^\s*[^:]+:\s*", "", anime_data[field]
                        ).strip()

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

                # Crawl screenshots using JS navigation
                print("Navigating to screenshots page...")
                js_navigate_screenshots = """
                const screenshotsLink = document.querySelector('a[href*="/screenshots"]');
                if (screenshotsLink) {
                    screenshotsLink.click();
                }
                """

                screenshots_raw_data = await _fetch_and_process_sub_page(
                    crawler,
                    url,
                    session_id,
                    js_navigate_screenshots,
                    "css:div#screenshots",
                    screenshots_css_schema,
                )

                if screenshots_raw_data and "screenshot_urls" in screenshots_raw_data:
                    anime_data["screenshots"] = [
                        item["url"]
                        for item in screenshots_raw_data["screenshot_urls"]
                        if "url" in item
                    ]
                    print(f"Extracted {len(anime_data['screenshots'])} screenshots")
                else:
                    anime_data["screenshots"] = []

                # Crawl relations using JS navigation with dropdown selection (two-step for reliability)
                print("Navigating to relations page...")

                # Step 1: Navigate to relations page
                js_navigate_relations = """
                const relationsLink = document.querySelector('a[href*="/relations"]');
                if (relationsLink) {
                    relationsLink.click();
                }
                """

                # Navigate first (no extraction yet)
                await _fetch_and_process_sub_page(
                    crawler,
                    url,
                    session_id,
                    js_navigate_relations,
                    "css:select.ofilter",
                    None,  # No extraction, just navigating
                )

                # Step 2: Select "overall" from dropdown and extract
                js_select_overall = """
                const dropdown = document.querySelector('select.ofilter[data-param="show"]');
                if (dropdown) {
                    dropdown.value = 'overall';
                    dropdown.dispatchEvent(new Event('change', { bubbles: true }));
                }
                """

                relations_raw_data = await _fetch_and_process_sub_page(
                    crawler,
                    url,
                    session_id,
                    js_select_overall,
                    "css:section#relations_anime tbody",
                    relations_css_schema,
                    use_js_only=True,
                )

                if relations_raw_data:
                    if "anime_relations" in relations_raw_data:
                        _process_relation_tooltips(
                            relations_raw_data["anime_relations"]
                        )
                        anime_data["anime_relations"] = relations_raw_data[
                            "anime_relations"
                        ]
                        print(
                            f"Extracted {len(anime_data['anime_relations'])} anime relations"
                        )
                    else:
                        anime_data["anime_relations"] = []

                    if "manga_relations" in relations_raw_data:
                        _process_relation_tooltips(
                            relations_raw_data["manga_relations"]
                        )
                        anime_data["manga_relations"] = relations_raw_data[
                            "manga_relations"
                        ]
                        print(
                            f"Extracted {len(anime_data['manga_relations'])} manga relations"
                        )
                    else:
                        anime_data["manga_relations"] = []
                else:
                    anime_data["anime_relations"] = []
                    anime_data["manga_relations"] = []

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
