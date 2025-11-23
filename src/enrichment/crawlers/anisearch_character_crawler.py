"""
Crawls character information from anisearch.com anime URLs with Redis caching.

Extracts character data using crawl4ai with CSS selectors, processes fields
like 'favorites' and 'image', and adds character roles. Results are cached
in Redis for 24 hours to avoid repeated crawling.

Usage:
    python -m src.enrichment.crawlers.anisearch_character_crawler <url> [--output PATH]

    <url>           anisearch.com anime character page URL
    --output PATH   optional output file path (default: anisearch_characters.json)
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Any, Dict, Optional

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from src.cache_manager.config import get_cache_config

logger = logging.getLogger(__name__)
from src.cache_manager.result_cache import cached_result
from src.enrichment.crawlers.utils import sanitize_output_path

# Get TTL from config to keep cache control centralized
_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

BASE_CHARACTER_URL = "https://www.anisearch.com/anime/"


def _normalize_character_url(anime_identifier: str) -> str:
    """
    Normalize an anime identifier into a full Anisearch characters page URL.
    
    Parameters:
        anime_identifier (str): A full Anisearch anime URL, a path, or a canonical anime id.
            Accepted forms:
            - Full URL with characters: "https://www.anisearch.com/anime/18878,dan-da-dan/characters"
            - Full URL without characters: "https://www.anisearch.com/anime/18878,dan-da-dan"
            - Path with or without leading slash: "/18878,dan-da-dan/characters" or "18878,dan-da-dan/characters"
            - Canonical anime id: "18878,dan-da-dan"
    
    Returns:
        str: A complete characters page URL, e.g. "https://www.anisearch.com/anime/18878,dan-da-dan/characters"
    """
    # If already full URL with /characters, return as-is
    if anime_identifier.startswith("https://www.anisearch.com/anime/") and "/characters" in anime_identifier:
        return anime_identifier

    # If full URL without /characters, append it
    if anime_identifier.startswith("https://www.anisearch.com/anime/"):
        return f"{anime_identifier.rstrip('/')}/characters"

    # Remove leading slash if present
    clean_id = anime_identifier.lstrip("/")

    # Remove /characters suffix if present (we'll add it back)
    clean_id = clean_id.replace("/characters", "").rstrip("/")

    # Construct full URL
    return f"{BASE_CHARACTER_URL}{clean_id}/characters"


def _extract_anime_id_from_character_url(url: str) -> str:
    """
    Extract the canonical anime identifier from an AniSearch character page URL.
    
    Parameters:
        url (str): Full character URL under BASE_CHARACTER_URL (e.g. "https://www.anisearch.com/anime/18878,dan-da-dan/characters").
    
    Returns:
        str: Canonical anime ID (e.g., "18878,dan-da-dan").
    
    Raises:
        ValueError: If the URL does not start with the expected base character URL.
    """
    # Remove base URL and /characters suffix
    if url.startswith(BASE_CHARACTER_URL):
        path = url[len(BASE_CHARACTER_URL):]
        # Remove /characters suffix
        return path.replace("/characters", "").rstrip("/")

    raise ValueError(f"Invalid character URL: {url}")


@cached_result(ttl=TTL_ANISEARCH, key_prefix="anisearch_characters")
async def _fetch_anisearch_characters_data(canonical_anime_id: str) -> Optional[Dict[str, Any]]:
    """
    Crawl anisearch.com and return normalized, enriched character data for the given canonical anime ID.
    
    Results are cached in Redis for 24 hours using the canonical anime ID as the cache key; this function performs no side effects.
    
    Parameters:
        canonical_anime_id (str): Canonical anime identifier already normalized (e.g., "18878,dan-da-dan").
    
    Returns:
        dict: A dictionary with a "characters" key containing a list of character objects enriched with
        `role`, absolute `url`, optional integer `favorites`, and `image` URL; or `None` if fetching or extraction failed.
    """
    # Build URL from canonical anime ID (caller already normalized)
    url = f"{BASE_CHARACTER_URL}{canonical_anime_id}/characters"
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

    extraction_strategy = JsonCssExtractionStrategy(css_schema)
    config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        page_timeout=60000,  # 60 seconds for image loading
        wait_until="networkidle",
        wait_for_images=True,
        scan_full_page=True,
        adjust_viewport_to_content=True,
        delay_before_return_html=2.0,  # 2 second delay for reliable image loading
    )

    async with AsyncWebCrawler() as crawler:
        results: RunManyReturn = await crawler.arun(url, config=config)

        if not results:
            logger.info("No results found.")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            logger.info(f"URL: {result.url}")
            logger.info(f"Success: {result.success}")

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
                        image_style = character.get("image")
                        if image_style:
                            match = re.search(
                                r'url\(["\\]?(.*?)["\\]?\)',
                                image_style,  # Corrected escaping here
                            )
                            if match:
                                character["image"] = match.group(1)

                        # Add role directly into the character object
                        character["role"] = role
                        character["url"] = (
                            "https://www.anisearch.com/" + character["url"]
                        )
                        flattened_characters.append(character)

                output_data = {"characters": flattened_characters}

                # Always return data (no conditional return or file writing)
                return output_data
            else:
                logger.warning(f"Extraction failed: {result.error_message}")
                return None


async def fetch_anisearch_characters(
    anime_id: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch character data for an anime from AniSearch, optionally save it to a file, and return the processed data.
    
    Parameters:
        anime_id (str): Anime identifier in one of three forms: a full characters URL (https://www.anisearch.com/anime/.../characters), a path (with or without a leading slash, e.g., "/18878,dan-da-dan/characters"), or a canonical ID without the "/characters" suffix (e.g., "18878,dan-da-dan").
        return_data (bool): If True, return the fetched character data; if False, do not return it (default: True).
        output_path (Optional[str]): If provided, write the resulting JSON to this file path (UTF-8, pretty-printed).
    
    Returns:
        dict: The complete processed character data dictionary when `return_data` is True; `None` otherwise.
    """
    # Normalize identifier once so cache keys depend on canonical anime ID
    # This ensures cache reuse across different identifier formats
    character_url = _normalize_character_url(anime_id)
    canonical_anime_id = _extract_anime_id_from_character_url(character_url)

    # Fetch data from cache or crawl (pure function keyed only on canonical anime ID)
    data = await _fetch_anisearch_characters_data(canonical_anime_id)

    if data is None:
        return None

    # Side effect: Write to file (always executes, even on cache hit)
    if output_path:
        safe_path = sanitize_output_path(output_path)
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data written to {safe_path}")

    # Return data based on return_data parameter
    if return_data:
        return data

    return None


async def main() -> int:
    """
    Run the CLI for crawling character data from anisearch.com and handle exit codes.
    
    Parses command-line arguments, invokes the crawler with the provided anime identifier, optionally writes output to a file, and returns an exit code.
    
    Returns:
        int: Exit code `0` on success, `1` if a parsing or runtime error occurred.
    """
    parser = argparse.ArgumentParser(
        description="Crawl character data from anisearch.com."
    )
    parser.add_argument(
        "anime_id",
        type=str,
        help="Anime identifier: full URL, path (e.g., '/18878,dan-da-dan/characters'), or ID (e.g., '18878,dan-da-dan')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anisearch_characters.json",
        help="Output file path (default: anisearch_characters.json in current directory)",
    )
    args = parser.parse_args()

    try:
        await fetch_anisearch_characters(
            args.anime_id,
            output_path=args.output,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))