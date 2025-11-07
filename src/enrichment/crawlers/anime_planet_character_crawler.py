"""
This script crawls character information from anime-planet.com anime character pages.

It accepts a slug as a command-line argument and extracts comprehensive character data
including detailed enrichment from individual character pages using concurrent batch processing.

The extracted data is saved to 'animeplanet_characters.json' in the project root.

Usage:
    python anime_planet_character_crawler.py <slug>
    python anime_planet_character_crawler.py dandadan
"""

import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from src.cache_manager.result_cache import cached_result

from .utils import sanitize_output_path

BASE_URL = "https://www.anime-planet.com"


def _normalize_characters_url(anime_identifier: str) -> str:
    """
    Normalize various input formats to full anime-planet characters URL.

    Accepts:
        - Full URL: "https://www.anime-planet.com/anime/dandadan/characters"
        - Slug: "dandadan"
        - Path: "/anime/dandadan" or "/anime/dandadan/characters"

    Returns:
        Full URL: "https://www.anime-planet.com/anime/dandadan/characters"
    """
    if not anime_identifier.startswith("http"):
        # Remove leading slashes and "anime/" prefix if present
        clean_id = anime_identifier.lstrip("/")
        if clean_id.startswith("anime/"):
            clean_id = clean_id[6:]  # Remove "anime/" prefix
        # Remove "/characters" suffix if present (we'll add it back)
        clean_id = clean_id.rstrip("/").replace("/characters", "")
        url = f"{BASE_URL}/anime/{clean_id}/characters"
    else:
        url = anime_identifier
        # Ensure URL ends with /characters
        if not url.endswith("/characters"):
            url = url.rstrip("/") + "/characters"

    if not url.startswith(f"{BASE_URL}/anime/"):
        raise ValueError(
            f"Invalid URL: Must be an anime-planet.com anime URL. "
            f"Expected format: '{BASE_URL}/anime/<slug>/characters' or just '<slug>'"
        )

    return url


def _extract_slug_from_characters_url(url: str) -> str:
    """Extract slug from anime-planet characters URL."""
    # Extract slug from: https://www.anime-planet.com/anime/dandadan/characters
    import re

    match = re.search(r"/anime/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Could not extract slug from URL: {url}")
    return match.group(1)


@cached_result(ttl=86400, key_prefix="animeplanet_characters")  # 24 hours cache
async def fetch_animeplanet_characters(
    slug: str, return_data: bool = True, output_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Crawls and processes character data from anime-planet.com.
    Uses concurrent batch processing for character detail enrichment.

    Args:
        slug: Anime slug (e.g., "dandadan"), path (e.g., "/anime/dandadan/characters"),
              or full URL (e.g., "https://www.anime-planet.com/anime/dandadan/characters")
        return_data: Whether to return the data dict (default: True)
        output_path: Optional file path to save JSON (default: None)

    Returns:
        Complete character data dictionary with enriched details (if return_data=True), otherwise None
    """
    # Normalize URL and extract slug using helper functions
    characters_url = _normalize_characters_url(slug)
    slug = _extract_slug_from_characters_url(characters_url)

    # Helper function to get reusable character field schema
    def _get_character_fields_schema() -> List[Dict[str, Any]]:
        return [
            {"name": "name", "selector": "a.name", "type": "text"},
            {
                "name": "url",
                "selector": "a.name",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "image_src",
                "selector": "img",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "image_data_src",
                "selector": "img",
                "type": "attribute",
                "attribute": "data-src",
            },
            {
                "name": "tags_raw",
                "selector": ".tags a",
                "type": "list",
                "fields": [{"name": "tag", "type": "text"}],
            },
            # Voice actors by language
            {
                "name": "voice_actors_jp",
                "selector": ".tableActors .flagJP a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "voice_actors_us",
                "selector": ".tableActors .flagUS a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "voice_actors_es",
                "selector": ".tableActors .flagES a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "voice_actors_fr",
                "selector": ".tableActors .flagFR a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
        ]

    # CSS schema uses helper function to avoid field duplication
    list_schema = {
        "baseSelector": "body",
        "fields": [
            # Extract Main characters
            {
                "name": "main_characters",
                "selector": "h3.sub:first-of-type + table tr",
                "type": "nested_list",
                "fields": _get_character_fields_schema(),
            },
            # Extract Secondary characters
            {
                "name": "secondary_characters",
                "selector": "h3.sub:nth-of-type(2) + table tr",
                "type": "nested_list",
                "fields": _get_character_fields_schema(),
            },
            # Extract Minor characters
            {
                "name": "minor_characters",
                "selector": "h3.sub:nth-of-type(3) + table tr",
                "type": "nested_list",
                "fields": _get_character_fields_schema(),
            },
        ],
    }

    async with AsyncWebCrawler() as crawler:
        print(f"Fetching character list: {characters_url}")

        # Phase 1: Fetch character list page
        list_config = CrawlerRunConfig(
            extraction_strategy=JsonCssExtractionStrategy(list_schema)
        )
        results: RunManyReturn = await crawler.arun(
            url=characters_url, config=list_config
        )

        if not results:
            print(f"Failed to fetch character list: {characters_url}")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if not data:
                    print("Extraction returned empty data.")
                    continue

                # Phase 2: Process character list and assign roles
                characters_basic = _process_character_list(
                    data[0],
                )

                if not characters_basic:
                    print("No characters found.")
                    return None

                print(f"Found {len(characters_basic)} characters")

                # Phase 3: Prepare character detail URLs for concurrent enrichment
                character_detail_urls = []
                for char in characters_basic:
                    # Extract slug from URL using simple string operations
                    url = char.get("url", "")
                    if url and "/characters/" in url:
                        char_slug = (
                            url.split("/characters/")[1].split("?")[0].split("#")[0]
                        )
                        character_detail_urls.append(
                            f"{BASE_URL}/characters/{char_slug}"
                        )

                if not character_detail_urls:
                    print("No valid character URLs found.")
                    return None

                # Phase 4: CONCURRENT BATCH ENRICHMENT using arun_many()
                print(
                    f"Enriching {len(character_detail_urls)} characters concurrently..."
                )

                detail_schema = _get_character_detail_schema()
                detail_config = CrawlerRunConfig(
                    extraction_strategy=JsonCssExtractionStrategy(detail_schema),
                )

                # Concurrent batch fetch of all character detail pages
                list_results = await crawler.arun_many(
                    urls=character_detail_urls,
                    config=detail_config,
                )

                # Validate the results - arun_many returns List[CrawlResultContainer]
                if not list_results:
                    print("No results returned from batch character fetch")
                    return None

                if not isinstance(list_results, list):
                    print(
                        f"Unexpected return type from arun_many: {type(list_results)}"
                    )
                    return None

                # Unwrap CrawlResultContainer objects to get CrawlResult objects
                # arun_many() returns List[CrawlResultContainer], each wrapping a CrawlResult
                from crawl4ai.models import CrawlResultContainer

                unwrapped_results: List[CrawlResult] = []
                for container in list_results:
                    # CrawlResultContainer is iterable and yields CrawlResult objects
                    if isinstance(container, CrawlResultContainer):
                        for result in container:
                            if isinstance(result, CrawlResult):
                                unwrapped_results.append(result)
                    elif isinstance(container, CrawlResult):
                        # Handle direct CrawlResult (shouldn't happen but be safe)
                        unwrapped_results.append(container)

                if not unwrapped_results:
                    print("No valid CrawlResult objects found after unwrapping")
                    return None

                # Replace list_results with unwrapped results
                list_results = unwrapped_results

                # Phase 5: Merge enriched data
                # Create a mapping from character name to index for proper matching
                char_name_to_index = {
                    char["name"]: i for i, char in enumerate(characters_basic)
                }

                enriched_count = 0
                for detail_result in list_results:

                    if detail_result.success and detail_result.extracted_content:
                        try:
                            detail_data = json.loads(detail_result.extracted_content)
                            if detail_data:
                                # Extract name from detail page to match with basic list
                                detail_name = detail_data[0].get("name_h1", "").strip()

                                if detail_name in char_name_to_index:
                                    idx = char_name_to_index[detail_name]
                                    enriched_data = _process_character_details(
                                        detail_data[0]
                                    )
                                    characters_basic[idx].update(enriched_data)
                                    enriched_count += 1
                                else:
                                    print(
                                        f"Warning: Could not match character: {detail_name}"
                                    )
                        except (
                            json.JSONDecodeError,
                            KeyError,
                            IndexError,
                            TypeError,
                        ) as e:
                            print(f"Failed to process enrichment: {e}")
                    else:
                        print(
                            f"Failed to enrich detail page - {detail_result.error_message}"
                        )

                print(
                    f"Successfully enriched {enriched_count}/{len(characters_basic)} characters"
                )

                # Prepare output data
                output_data = {
                    "characters": characters_basic,
                    "total_count": len(characters_basic),
                }

                # Conditionally write to file
                if output_path:
                    safe_path = sanitize_output_path(output_path)
                    with open(safe_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                    print(f"Data written to {safe_path}")

                # Return data for programmatic usage
                if return_data:
                    return output_data

                return None
            else:
                print(f"Extraction failed: {result.error_message}")
                return None
        return None


def _get_character_detail_schema() -> Dict[str, Any]:
    """Get CSS schema for character detail page extraction.

    Note: We don't extract tables here since CSS can't easily get
    tables following h3 headers. We'll parse from raw HTML instead.
    """
    return {
        "baseSelector": "body",
        "fields": [
            # Basic info
            {"name": "name_h1", "selector": "h1[itemprop='name']", "type": "text"},
            {
                "name": "image_detail",
                "selector": "img[itemprop='image']",
                "type": "attribute",
                "attribute": "src",
            },
            # Entry bar data (contains gender, hair color, ranks)
            {
                "name": "entry_bar_items",
                "selector": ".entryBar .pure-1",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            # Loved rank
            {
                "name": "loved_rank",
                "selector": ".entryBar .fa-heart + a",
                "type": "text",
            },
            # Hated rank
            {
                "name": "hated_rank",
                "selector": ".entryBar .heartOff + a",
                "type": "text",
            },
            # Metadata section (eye color, age, birthday, etc.)
            {
                "name": "metadata_items",
                "selector": ".EntryMetadata__item",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "title",
                        "selector": ".EntryMetadata__title",
                        "type": "text",
                    },
                    {
                        "name": "value",
                        "selector": ".EntryMetadata__value",
                        "type": "text",
                    },
                ],
            },
            # Description
            {
                "name": "description_paragraphs",
                "selector": ".entrySynopsis p",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            # Tags
            {
                "name": "tags_detail",
                "selector": ".tags a",
                "type": "list",
                "fields": [{"name": "tag", "type": "text"}],
            },
            # Alternative names
            {
                "name": "alt_names_raw",
                "selector": ".entryAltNames li",
                "type": "list",
                "fields": [{"name": "name", "type": "text"}],
            },
            # Anime roles - extract table rows following "Anime Roles" header
            {
                "name": "anime_roles_raw",
                "selector": "h3:contains('Anime Roles') + table tr",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "anime_title",
                        "selector": "td:nth-child(1) a",
                        "type": "text",
                    },
                    {
                        "name": "anime_url",
                        "selector": "td:nth-child(1) a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {"name": "role", "selector": "td:nth-child(2)", "type": "text"},
                ],
            },
            # Manga roles - extract table rows following "Manga Roles" header
            {
                "name": "manga_roles_raw",
                "selector": "h3:contains('Manga Roles') + table tr",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "manga_title",
                        "selector": "td:nth-child(1) a",
                        "type": "text",
                    },
                    {
                        "name": "manga_url",
                        "selector": "td:nth-child(1) a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {"name": "role", "selector": "td:nth-child(2)", "type": "text"},
                ],
            },
        ],
    }


def _process_character_list(list_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process character list data and assign roles based on section headers.

    Args:
        list_data: Extracted data with main_characters, secondary_characters, minor_characters

    Returns:
        Flat list of character dictionaries with role assigned
    """
    characters = []

    # Process each role section separately
    role_sections = {
        "Main": list_data.get("main_characters", []),
        "Secondary": list_data.get("secondary_characters", []),
        "Minor": list_data.get("minor_characters", []),
    }

    for role, character_rows in role_sections.items():
        for row in character_rows:
            name = row.get("name", "").strip()
            url = row.get("url", "").strip()

            if not name or not url:
                continue

            char_data = {
                "name": name,
                "url": url,
                "role": role,  # Role is determined by which section it came from
            }

            # Handle image (src or data-src fallback)
            image = row.get("image_src") or row.get("image_data_src")
            if image:
                char_data["image"] = image

            # Parse tags
            tags_raw = row.get("tags_raw", [])
            if tags_raw:
                tags = [
                    tag.get("tag", "").strip() for tag in tags_raw if tag.get("tag")
                ]
                if tags:
                    char_data["tags"] = tags

            # Parse voice actors using helper function
            voice_actors = _extract_voice_actors(row)
            if voice_actors:
                char_data["voice_actors"] = voice_actors

            characters.append(char_data)

    return characters


def _extract_voice_actors(character: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Extract voice actors by language from character data.

    Args:
        character: Character dictionary containing voice actor data

    Returns:
        Dictionary mapping language codes to voice actor lists
    """
    voice_actors: Dict[str, List[Dict[str, str]]] = {}

    for lang_code in ["jp", "us", "es", "fr"]:
        va_list = character.get(f"voice_actors_{lang_code}", [])
        if va_list:
            voice_actors[lang_code] = [
                {"name": va.get("name", "").strip(), "url": va.get("url", "")}
                for va in va_list
                if va.get("name")
            ]

    return voice_actors


def _normalize_value(value: str) -> Optional[str]:
    """Convert '?' to None, otherwise return stripped value."""
    stripped = value.strip()
    return None if stripped == "?" else stripped


def _process_character_details(detail_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process character detail page data into enriched character info.

    Args:
        detail_data: Extracted data from CSS schema
        html: Raw HTML from detail page for table parsing

    Returns:
        Dictionary of enriched character data
    """
    enriched: Dict[str, Any] = {}

    # Extract gender and hair color from entry bar
    entry_bar_items = detail_data.get("entry_bar_items", [])

    for item in entry_bar_items:
        text = item.get("text", "")

        if "Gender:" in text:
            gender_value = text.replace("Gender:", "").strip()
            enriched["gender"] = _normalize_value(gender_value)

        if "Hair Color:" in text:
            hair_value = text.replace("Hair Color:", "").strip()
            enriched["hair_color"] = _normalize_value(hair_value)

    # Extract loved/hated ranks from CSS
    loved_rank_text = detail_data.get("loved_rank", "").strip()
    if loved_rank_text:
        # Remove #, commas and convert to int
        try:
            enriched["loved_rank"] = int(
                loved_rank_text.replace("#", "").replace(",", "")
            )
        except ValueError:
            pass

    hated_rank_text = detail_data.get("hated_rank", "").strip()
    if hated_rank_text:
        try:
            enriched["hated_rank"] = int(
                hated_rank_text.replace("#", "").replace(",", "")
            )
        except ValueError:
            pass

    # Extract metadata (eye_color, age, birthday, etc.)
    metadata_items = detail_data.get("metadata_items", [])
    for item in metadata_items:
        title = item.get("title", "").strip()
        value = item.get("value", "").strip()

        if title and value:
            # Convert "Eye Color" -> "eye_color"
            field_name = title.lower().replace(" ", "_")
            enriched[field_name] = _normalize_value(value)

    # Extract description
    desc_paragraphs = detail_data.get("description_paragraphs", [])
    for p in desc_paragraphs:
        text = p.get("text", "").strip()
        if text and len(text) > 50 and "Tags" not in text:
            enriched["description"] = text
            break

    # Extract alternative names
    alt_names_raw = detail_data.get("alt_names_raw", [])
    if alt_names_raw:
        alt_names = [
            item.get("name", "").strip() for item in alt_names_raw if item.get("name")
        ]
        if alt_names:
            enriched["alternative_names"] = alt_names

    # Parse anime roles from CSS-extracted data
    anime_roles_raw = detail_data.get("anime_roles_raw", [])
    if anime_roles_raw:
        anime_roles = []
        for role_data in anime_roles_raw:
            anime_title = role_data.get("anime_title", "").strip()
            anime_url = role_data.get("anime_url", "").strip()
            role = role_data.get("role", "").strip()

            if anime_title and anime_url:
                anime_role = {
                    "anime_title": anime_title,
                    "anime_url": anime_url,
                }
                # Add role if present (Main, Secondary, Minor, etc.)
                if role:
                    anime_role["role"] = role
                anime_roles.append(anime_role)

        if anime_roles:
            enriched["anime_roles"] = anime_roles

    # Parse manga roles from CSS-extracted data
    manga_roles_raw = detail_data.get("manga_roles_raw", [])
    if manga_roles_raw:
        manga_roles = []
        for role_data in manga_roles_raw:
            manga_title = role_data.get("manga_title", "").strip()
            manga_url = role_data.get("manga_url", "").strip()
            role = role_data.get("role", "").strip()

            if manga_title and manga_url:
                manga_role = {
                    "manga_title": manga_title,
                    "manga_url": manga_url,
                }
                # Add role if present (Main, Secondary, Minor, etc.)
                if role:
                    manga_role["role"] = role
                manga_roles.append(manga_role)

        if manga_roles:
            enriched["manga_roles"] = manga_roles

    return enriched


async def main() -> int:
    """CLI entry point for anime-planet.com character crawler."""
    parser = argparse.ArgumentParser(
        description="Crawl character data from an anime-planet.com anime characters page."
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="Anime identifier: slug (e.g., 'dandadan'), path (e.g., '/anime/dandadan/characters'), or full URL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="animeplanet_characters.json",
        help="Output file path (default: animeplanet_characters.json in current directory)",
    )
    args = parser.parse_args()

    try:
        await fetch_animeplanet_characters(
            args.identifier,
            return_data=False,  # CLI doesn't need return value
            output_path=args.output,
        )
        return 0
    except Exception as e:
        import sys
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(asyncio.run(main()))
