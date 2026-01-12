"""AniDB Character Crawler for extracting character metadata.

This crawler fetches detailed character information from AniDB character pages.
It extracts comprehensive character metadata including names, abilities,
personality, appearance, roles, and ratings.

Uses crawl4ai with realistic browser headers and stealth configuration to bypass
AniDB's anti-leech protection. No UndetectedAdapter required.

Usage:
    ./pants run libs/enrichment/src/enrichment/crawlers/anidb_character_crawler.py -- <character_id> [--output PATH]

    <character_id>  AniDB character ID (e.g., 491 for Brook)
    --output PATH   optional output file path (default: anidb_character.json)

Example:
    >>> from enrichment.crawlers.anidb_character_crawler import fetch_anidb_character
    >>> data = await fetch_anidb_character(491)  # Brook from One Piece
    >>> data = await fetch_anidb_character(491, output_path="brook.json")
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn
from enrichment.crawlers.utils import sanitize_output_path
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

# Get TTL from centralized config
_CACHE_CONFIG = get_cache_config()
TTL_ANIDB_CHARACTER = _CACHE_CONFIG.ttl_anidb

BASE_URL = "https://anidb.net/character"


def _get_character_schema() -> dict[str, Any]:
    """Get the CSS extraction schema for AniDB character pages.

    Returns:
        Schema dictionary for JsonCssExtractionStrategy containing field
            definitions for character data extraction.
    """
    return {
        "description": "CSS extraction schema for AniDB character pages",
        "baseSelector": "body",
        "fields": [
            {
                "name": "name_main",
                "selector": "#tab_1_pane tr.mainname td.value span[itemprop='name']",
                "type": "text",
            },
            {
                "name": "name_kanji",
                "selector": "#tab_1_pane tr.official.verified.yes td.value label[itemprop='alternateName']",
                "type": "text",
            },
            {
                "name": "nicknames",
                "selector": "#tab_2_pane tr.nick td.value",
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


def _flatten_character_data(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested list fields from crawl4ai output.

    Converts [{"text": "value1"}, {"text": "value2"}] to ["value1", "value2"]
    for all list fields in the character data.

    Args:
        data: Raw character data dictionary from crawler with nested list
            structures.

    Returns:
        New dictionary with flattened character data with simple string arrays
            instead of nested dictionaries. Input dictionary is not modified.
    """
    # Create a shallow copy to avoid mutating the input
    flattened = data.copy()

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
        if field in flattened and isinstance(flattened[field], list):
            flattened[field] = [
                obj["text"] for obj in flattened[field] if "text" in obj
            ]

    return flattened


@cached_result(ttl=TTL_ANIDB_CHARACTER, key_prefix="anidb_character")
async def _fetch_anidb_character_data(
    canonical_character_id: int,
) -> dict[str, Any] | None:
    """Perform actual AniDB character crawling with caching.

    Pure function with no side effects - cached by character_id in Redis.
    Schema hash auto-invalidates cache on code changes.

    Args:
        canonical_character_id: AniDB character ID (e.g., 491 for Brook).

    Returns:
        Character data dictionary containing name, gender, abilities, looks,
            personality, and role information if successful, None otherwise.
    """
    url = f"{BASE_URL}/{canonical_character_id}"

    # Configure browser with stealth + realistic headers to bypass bot detection
    # Note: UndetectedAdapter is NOT required - headers alone are sufficient
    # headless=True works with our header config (no browser popup)
    browser_config = BrowserConfig(
        enable_stealth=True,
        headless=True,
        verbose=True,
        headers={
            # Critical: Realistic browser headers bypass AniDB's 403 bot detection
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        viewport_width=1920,
        viewport_height=1080,
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
        wait_until="domcontentloaded",
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results: RunManyReturn = await crawler.arun(url, config=config)

        if not results:
            logger.error(f"No results returned for character {canonical_character_id}")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            # Check both success flag AND status code (success can be True even with 403)
            status_code = result.status_code if hasattr(result, "status_code") else None

            # Validate we got a successful HTTP response
            if not result.success:
                error_msg = (
                    result.error_message
                    if hasattr(result, "error_message")
                    else "Unknown error"
                )
                logger.error(
                    f"Failed to fetch character {canonical_character_id}: {error_msg}"
                )
                return None

            if status_code and status_code != 200:
                logger.error(
                    f"HTTP error {status_code} for character {canonical_character_id}"
                )

                # Check for specific bot detection responses
                if status_code == 403:
                    if result.html and "AntiLeech" in result.html:
                        logger.error(
                            "Hit AniDB AntiLeech protection. You may be temporarily banned. "
                            "Wait 15 minutes to 24 hours before retrying."
                        )
                    else:
                        logger.error(
                            "403 Forbidden - Bot detection triggered. "
                            "Check browser headers and anti-detection settings."
                        )

                return None

            if result.extracted_content:
                # Parse JSON response
                data_list = json.loads(result.extracted_content)

                if not data_list or len(data_list) == 0:
                    logger.warning(
                        f"No character data extracted for ID {canonical_character_id}"
                    )
                    return None

                # Get first item (should only be one)
                character_data = data_list[0]

                # Flatten nested list fields
                character_data = _flatten_character_data(character_data)

                # Return pure data (no side effects)
                return character_data

            # If we reach here, success=True and status=200 but no extracted content
            logger.warning(
                f"No content extracted for character {canonical_character_id} "
                "despite successful crawl. Check CSS selectors."
            )
            return None

    return None


async def fetch_anidb_character(
    character_id: int,
    output_path: str | None = None,
) -> dict[str, Any] | None:
    """Fetch AniDB character data by character ID.

    Public API to fetch character information from AniDB character pages.
    Optionally saves the result to a JSON file.

    Args:
        character_id: AniDB character ID (e.g., 491 for Brook).
        output_path: Optional file path to save JSON output. If provided, the
            character data will be written to this path.

    Returns:
        Character data dictionary if successful, None otherwise.

    Example:
        >>> data = await fetch_anidb_character(491)
        >>> print(data['name_kanji'])
        ブルック
    """
    # Call cached function
    data = await _fetch_anidb_character_data(character_id)

    if data is None:
        return None

    # Side effect: write to file
    if output_path:
        safe_path = sanitize_output_path(output_path)
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data written to {safe_path}")

    return data


async def main() -> int:
    """Run CLI to crawl AniDB character page.

    Parses command-line arguments for character ID and output path,
    then fetches and saves character data.

    Returns:
        Exit code where 0 indicates success and 1 indicates failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Crawl character data from AniDB")
    parser.add_argument(
        "character_id",
        type=int,
        help="AniDB character ID (e.g., 491 for Brook)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anidb_character.json",
        help="Output file path (default: anidb_character.json)",
    )
    args = parser.parse_args()

    try:
        data = await fetch_anidb_character(
            args.character_id,
            output_path=args.output,
        )
        if data is None:
            logger.error("No data was extracted; see logs above for details.")
            return 1
    except (ValueError, OSError):
        logger.exception("Failed to fetch AniDB character data")
        return 1
    except Exception:
        logger.exception("Unexpected error during character fetch")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
