"""
AniDB Character Crawler

This crawler fetches detailed character information from AniDB character pages.
It extracts comprehensive character metadata including names, abilities, personality,
appearance, roles, and ratings.

Uses crawl4ai with UndetectedAdapter to bypass AniDB's anti-leech protection.

Usage:
    from enrichment.crawlers.anidb_character_crawler import fetch_anidb_character

    # Fetch character by ID
    data = await fetch_anidb_character(491)  # Brook from One Piece

    # Save to file
    data = await fetch_anidb_character(491, output_path="brook.json")
"""

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlResult,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    UndetectedAdapter,
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.types import RunManyReturn

logger = logging.getLogger(__name__)

BASE_URL = "https://anidb.net/character"


def _get_character_schema() -> Dict[str, Any]:
    """
    Get the CSS extraction schema for AniDB character pages.

    Returns:
        Schema dictionary for JsonCssExtractionStrategy
    """
    return {
        "description": "CSS extraction schema for AniDB character pages",
        "baseSelector": "body",
        "fields": [
            {
                "name": "name_kanji",
                "selector": "#tab_1_pane tr.official.verified.yes td.value label[itemprop='alternateName']",
                "type": "text",
            },
            {
                "name": "nicknames",
                "selector": "#tab_2_pane tr.alias td.value",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            {
                "name": "official_names",
                "selector": "#tab_2_pane tr.official td.value label[itemprop='alternateName']",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            {
                "name": "gender",
                "selector": "#tab_1_pane tr.gender td.value span[itemprop='gender']",
                "type": "text",
            },
            {
                "name": "abilities",
                "selector": "#tab_1_pane tr.abilities td.value span.tagname",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            {
                "name": "looks",
                "selector": "#tab_1_pane tr.looks td.value span.tagname",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            {
                "name": "personality",
                "selector": "#tab_1_pane tr.personality td.value span.tagname",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            {
                "name": "role",
                "selector": "#tab_1_pane tr.role td.value span.tagname",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            {
                "name": "supernatural_abilities",
                "selector": "#tab_1_pane tr[class*='supernatural'] td.value span.tagname",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
        ],
    }


def _flatten_character_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested list fields from crawl4ai output.

    Converts [{"text": "value1"}, {"text": "value2"}] to ["value1", "value2"]

    Args:
        data: Raw character data from crawler

    Returns:
        Flattened character data with simple arrays
    """
    list_fields = [
        "nicknames",
        "official_names",
        "abilities",
        "looks",
        "personality",
        "role",
        "supernatural_abilities",
    ]

    for field in list_fields:
        if field in data and isinstance(data[field], list):
            data[field] = [obj["text"] for obj in data[field] if "text" in obj]

    return data


async def fetch_anidb_character(
    character_id: int,
    return_data: bool = True,
    output_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch character information from AniDB by character ID.

    Uses UndetectedAdapter to bypass AniDB's anti-leech protection.

    Args:
        character_id: AniDB character ID (e.g., 491 for Brook)
        return_data: Whether to return the data dict (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        Character data dictionary (if return_data=True), otherwise None

    Example:
        >>> data = await fetch_anidb_character(491)
        >>> print(data['name_kanji'])
        ブルック
    """
    url = f"{BASE_URL}/{character_id}"

    logger.info(f"Fetching AniDB character: {url}")

    # Configure browser with advanced anti-bot bypass
    browser_config = BrowserConfig(
        enable_stealth=True,
        headless=True,
        verbose=True,
    )

    undetected_adapter = UndetectedAdapter()
    crawler_strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=browser_config,
        browser_adapter=undetected_adapter,
    )

    # Configure extraction with anti-detection features
    schema = _get_character_schema()
    extraction_strategy = JsonCssExtractionStrategy(schema)
    config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        delay_before_return_html=1.0,  # Wait 1 second before capturing
        simulate_user=True,  # Simulate mouse movements
        magic=True,  # Auto-handle popups and consent banners
        override_navigator=True,  # Override navigator properties for stealth
        mean_delay=2.0,  # Random delays
        max_range=1.0,
    )

    async with AsyncWebCrawler(
        crawler_strategy=crawler_strategy,
        config=browser_config,
    ) as crawler:
        logger.info(f"Starting crawl for character {character_id}...")

        results: RunManyReturn = await crawler.arun(url, config=config)

        if not results:
            logger.error(f"No results returned for character {character_id}")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            logger.info(f"Crawl status: {result.success}")

            if result.success and result.extracted_content:
                # Parse JSON response
                data_list = json.loads(result.extracted_content)

                if not data_list or len(data_list) == 0:
                    logger.warning(f"No character data extracted for ID {character_id}")
                    return None

                # Get first item (should only be one)
                character_data = data_list[0]

                # Flatten nested list fields
                character_data = _flatten_character_data(character_data)

                logger.info(f"Successfully extracted character data for {character_id}")
                logger.info(f"Name: {character_data.get('name_kanji')}")
                logger.info(f"Gender: {character_data.get('gender')}")

                # Save to file if requested
                if output_path:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(character_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Character data saved to {output_path}")

                return character_data if return_data else None

            else:
                error_msg = (
                    result.error_message
                    if hasattr(result, "error_message")
                    else "Unknown error"
                )
                logger.error(f"Failed to fetch character {character_id}: {error_msg}")

                # Check if we hit anti-leech
                if result.html and "AntiLeech" in result.html:
                    logger.error(
                        "Hit AniDB AntiLeech protection. You may be temporarily banned. "
                        "Wait 15 minutes to 24 hours before retrying."
                    )

                return None

    return None


async def main() -> None:
    """CLI entry point for testing the crawler."""
    parser = argparse.ArgumentParser(
        description="Fetch character data from AniDB by character ID"
    )
    parser.add_argument(
        "character_id",
        type=int,
        help="AniDB character ID (e.g., 491 for Brook)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output JSON file path (optional)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Fetch character data
    data = await fetch_anidb_character(
        character_id=args.character_id,
        output_path=args.output,
    )

    if data:
        print("\n=== CHARACTER DATA ===")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"Failed to fetch character {args.character_id}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
