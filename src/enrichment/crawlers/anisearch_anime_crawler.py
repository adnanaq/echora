"""
Crawls anime information from anisearch.com URLs with Redis caching.

Extracts comprehensive anime data including metadata, screenshots, and relations
using crawl4ai with CSS selectors and JavaScript navigation. Results are cached
in Redis for 24 hours to avoid repeated crawling.

Usage:
    python -m src.enrichment.crawlers.anisearch_anime_crawler <url> [--output PATH]

    <url>           anisearch.com anime page URL (full or relative path)
    --output PATH   optional output file path (default: anisearch_anime.json)
"""

import argparse
import asyncio
import html  # Import the html module for unescaping HTML entities
import json
import logging
import re
import sys
import uuid
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
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch


def _process_relation_tooltips(relations_list: List[Dict[str, Any]]) -> None:
    """
    Processes a list of relations to extract image URLs from tooltip_html and renames the field.
    """
    for relation in relations_list:
        image = relation.get("image")
        if image:
            unescaped_html = html.unescape(image)
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
                return cast(Dict[str, Any], sub_page_data[0])
    return None


BASE_ANIME_URL = "https://www.anisearch.com/anime/"

# Error messages
_ERR_INVALID_URL = (
    "Invalid URL: Must be an anisearch.com anime URL. "
    "Expected format: '{base_url}<anime-id>' or '/<anime-id>' or '<anime-id>'"
)
_ERR_URL_PREFIX = "URL must start with {base_url}"
_ERR_MISSING_PATH = "URL does not contain anime path"


def _normalize_anime_url(anime_identifier: str) -> str:
    """
    Normalize various input formats to full anisearch anime URL.

    Accepts:
        - Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan"
        - Relative path: "/18878,dan-da-dan" or "18878,dan-da-dan"

    Returns:
        Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan"

    Raises:
        ValueError: If URL is not from anisearch.com/anime
    """
    # Normalize the URL
    if not anime_identifier.startswith("http"):
        # Remove leading slash if present, then construct full URL
        url = f"{BASE_ANIME_URL}{anime_identifier.lstrip('/')}"
    else:
        url = anime_identifier

    # Validate it's an anisearch.com anime URL
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(_ERR_INVALID_URL.format(base_url=BASE_ANIME_URL))

    return url


def _extract_path_from_url(url: str) -> str:
    """
    Extract anime path from anisearch URL.

    Args:
        url: Full anisearch anime URL

    Returns:
        Anime path (e.g., "18878,dan-da-dan")

    Raises:
        ValueError: If URL cannot be parsed
    """
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(_ERR_URL_PREFIX.format(base_url=BASE_ANIME_URL))

    # Extract path after BASE_ANIME_URL
    path = url[len(BASE_ANIME_URL):]
    if not path:
        raise ValueError(_ERR_MISSING_PATH)

    return path


@cached_result(ttl=TTL_ANISEARCH, key_prefix="anisearch_anime")
async def _fetch_anisearch_anime_data(canonical_path: str) -> Optional[Dict[str, Any]]:
    """
    Pure cached function that crawls and processes anime data from anisearch.com.
    Uses JS-based navigation for screenshots and relations pages.

    Results are cached in Redis for 24 hours based ONLY on canonical path.
    This function has no side effects - it only fetches and returns data.

    Args:
        canonical_path: Canonical anime path (e.g., "18878,dan-da-dan") - already normalized by caller

    Returns:
        Complete anime data dictionary with enriched details, or None if fetch fails
    """
    # Build URL from canonical path (caller already normalized)
    url = f"{BASE_ANIME_URL}{canonical_path}"

    # Generate unique session ID for maintaining browser state

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

        logging.info(f"Fetching main page: {url}")
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
                data = cast(List[Dict[str, Any]], json.loads(result.extracted_content))

                if not data or not isinstance(data, list) or len(data) == 0:
                    logging.warning("Extraction returned empty data.")
                    return None

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
                published_str = anime_data.get("published")
                if published_str and isinstance(published_str, str):
                    match = re.search(
                        r"(\d{2}\.\d{2}\.\d{4})\s*[-–]\s*(\d{2}\.\d{2}\.\d{4})",
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
                logging.info("Navigating to screenshots page...")
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
                    logging.info(f"Extracted {len(anime_data['screenshots'])} screenshots")
                else:
                    anime_data["screenshots"] = []

                # Crawl relations using JS navigation with dropdown selection (two-step for reliability)
                logging.info("Navigating to relations page...")

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
                        logging.info(
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
                        logging.info(
                            f"Extracted {len(anime_data['manga_relations'])} manga relations"
                        )
                    else:
                        anime_data["manga_relations"] = []
                else:
                    anime_data["anime_relations"] = []
                    anime_data["manga_relations"] = []

                # Always return data (no conditional return or file writing)
                return anime_data

            # Log extraction failure but continue to check remaining results
            logging.warning(f"Extraction failed: {result.error_message}")

        # If loop completes without returning, no valid results were found
        return None


async def fetch_anisearch_anime(
    anime_id: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Wrapper function that handles side effects (file writing, return_data logic).

    This function calls the cached _fetch_anisearch_anime_data() to get the data,
    then performs side effects that should execute regardless of cache status.

    Args:
        anime_id: Anime identifier - can be:
            - Full URL: "https://www.anisearch.com/anime/18878,dan-da-dan"
            - Relative path: "/18878,dan-da-dan"
            - Anime ID: "18878,dan-da-dan"
        return_data: Whether to return the data dict (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        Complete anime data dictionary (if return_data=True), otherwise None
    """
    # Normalize identifier once so cache keys depend on canonical path
    # This ensures cache reuse across different identifier formats
    anime_url = _normalize_anime_url(anime_id)
    canonical_path = _extract_path_from_url(anime_url)

    # Fetch data from cache or crawl (pure function keyed only on canonical path)
    data = await _fetch_anisearch_anime_data(canonical_path)

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


async def main() -> int:
    """CLI entry point for anisearch.com anime crawler."""
    parser = argparse.ArgumentParser(
        description="Crawl anime data from anisearch.com anime page."
    )
    parser.add_argument(
        "anime_id",
        type=str,
        help="Anime identifier: full URL, path (e.g., '/18878,dan-da-dan'), or ID (e.g., '18878,dan-da-dan')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anisearch_anime.json",
        help="Output file path (default: anisearch_anime.json in current directory)",
    )
    args = parser.parse_args()

    try:
        await fetch_anisearch_anime(
            args.anime_id,
            output_path=args.output,
        )
    except (ValueError, OSError):
        logging.exception("Failed to fetch anisearch anime data")
        return 1
    except Exception:
        logging.exception("Unexpected error during anime fetch")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
