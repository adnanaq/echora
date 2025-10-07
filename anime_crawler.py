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
import asyncio
import json
import re
import argparse
import html # Import the html module for unescaping HTML entities
from typing import cast, List, Dict, Any

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn


async def fetch_anisearch_anime(url: str):
    """
    Crawls, processes, and saves anime data from a given anisearch.com URL.

    This function defines a schema for extracting anime information,
    processes the returned data to clean it, and finally writes the output
    to a JSON file.

    Args:
        url (str): The URL of the anisearch.com anime page to crawl.
    """
    css_schema = {
        "baseSelector": "body", # Changed baseSelector to body to encompass all sections
        "fields": [
            { "name": "cover_image", "selector": "section#information img#details-cover", "type": "attribute", "attribute": "src" },
            { "name": "japanese_title", "selector": "section#information div.title[lang='ja'] strong.f16", "type": "text" },
            { "name": "japanese_title_alt", "selector": "section#information div.title[lang='ja'] div.grey", "type": "text" },
            { "name": "type", "selector": "section#information div.type", "type": "text" },
            { "name": "status", "selector": "section#information div.status", "type": "text" },
            { "name": "published", "selector": "section#information div.released", "type": "text" },
            { "name": "studio", "selector": "section#information div.company a[href*='company']", "type": "text" },
            {
                "name": "staff",
                "selector": "section#information div.creators a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"}
                ]
            },
            { "name": "source_material", "selector": "section#information div.adapted", "type": "text" },
            {
                "name": "websites",
                "selector": "section#information div.websites a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"}
                ]
            },
            { "name": "english_title", "selector": "section#information div.title[lang='en'] strong.f16", "type": "text" },
            { "name": "english_status", "selector": "section#information ul.xlist > li:nth-child(2) div.status", "type": "text" },
            { "name": "english_published", "selector": "section#information ul.xlist > li:nth-child(2) div.released", "type": "text" },
            {
                "name": "publishers",
                "selector": "section#information ul.xlist > li:nth-child(2) div.company a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"}
                ]
            },
            { "name": "synonyms", "selector": "section#information div.synonyms", "type": "text" },
            { "name": "description", "selector": "section#description div.textblock.details-text", "type": "text" },
            {
                "name": "genres",
                "selector": "section#genres-tags ul.cloud a.gg, section#genres-tags ul.cloud a.gc",
                "type": "list",
                "fields": [{"name": "genre", "type": "text"}]
            },
            {
                "name": "tags",
                "selector": "section#genres-tags ul.cloud a.gt",
                "type": "list",
                "fields": [{"name": "tag", "type": "text"}]
            },
            { "name": "screenshots_page_url", "selector": "section#images div.showall a.sbuttonA", "type": "attribute", "attribute": "href" },
            { "name": "relations_page_url", "selector": "section#relations div.showall a.sbuttonA", "type": "attribute", "attribute": "href" }
        ]
    }

    async with AsyncWebCrawler() as crawler:
        # Define main extraction strategy and config inside the crawler context
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

                # Helper function to clean text fields
                def clean_field(text: str) -> str:
                    if text: # Check if text is not None or empty
                        # Remove header like "Type:", "Status:", "Published:", "Adapted From:", "Synonyms:"
                        # This regex matches any characters from the start of the string up to the first colon, followed by optional whitespace.
                        return re.sub(r'^\s*[^:]+:\s*', '', text).strip()
                    return text

                # Clean up the data
                for field in ["type", "status", "published", "source_material", "synonyms", "english_status", "english_published"]:
                    if field in anime_data and isinstance(anime_data[field], str):
                        anime_data[field] = clean_field(anime_data[field])

                # Extract start_date and end_date from published field
                anime_data["start_date"] = None
                anime_data["end_date"] = None
                if "published" in anime_data and anime_data["published"]:
                    published_str = anime_data["published"]
                    match = re.search(r'(\d{2}\.\d{2}\.\d{4})\s*‑\s*(\d{2}\.\d{2}\.\d{4})', published_str)
                    if match:
                        anime_data["start_date"] = match.group(1)
                        anime_data["end_date"] = match.group(2)
                    else:
                        # Handle case with only one date or different format
                        single_date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', published_str)
                        if single_date_match:
                            anime_data["start_date"] = single_date_match.group(0)
                            anime_data["end_date"] = None # Or same as start_date, depending on desired behavior

                # Remove the original 'published' field after parsing
                if "published" in anime_data:
                    del anime_data["published"]

                # Flatten genres
                if "genres" in anime_data and isinstance(anime_data["genres"], list):
                    anime_data["genres"] = [item["genre"] for item in anime_data["genres"] if "genre" in item]

                # Flatten tags
                if "tags" in anime_data and isinstance(anime_data["tags"], list):
                    anime_data["tags"] = [item["tag"] for item in anime_data["tags"] if "tag" in item]

                # Define screenshots schema and config inside the crawler context as well
                screenshots_css_schema = {
                    "baseSelector": "body", # Screenshots page might also have a body tag as base
                    "fields": [
                        {
                            "name": "screenshot_urls",
                            "selector": "div#screenshots a.zoom", # Corrected selector
                            "type": "nested_list", # Changed to nested_list
                            "fields": [
                                {"name": "url", "type": "attribute", "attribute": "href"}
                            ]
                        }
                    ]
                }
                screenshots_extraction_strategy = JsonCssExtractionStrategy(screenshots_css_schema)
                screenshots_config = CrawlerRunConfig(extraction_strategy=screenshots_extraction_strategy)

                # Extract screenshots from the linked page
                if "screenshots_page_url" in anime_data and anime_data["screenshots_page_url"]:
                    relative_url = anime_data["screenshots_page_url"]
                    full_screenshots_url = f"https://www.anisearch.com/{relative_url}"
                    
                    print(f"Fetching screenshots from: {full_screenshots_url}")
                    screenshots_results: RunManyReturn = await crawler.arun(full_screenshots_url, config=screenshots_config)

                    if screenshots_results:
                        for ss_result in screenshots_results:
                            ss_result = cast(CrawlResult, ss_result)
                            if ss_result.success and ss_result.extracted_content:
                                ss_data = json.loads(ss_result.extracted_content)
                                # ss_data will be a list of dictionaries, each containing a 'screenshot_urls' key
                                # We expect only one result from the screenshots page
                                if ss_data and "screenshot_urls" in ss_data[0]:
                                    # Flatten the list of screenshot objects
                                    anime_data["screenshots"] = [item["url"] for item in ss_data[0]["screenshot_urls"] if "url" in item]
                                else:
                                    anime_data["screenshots"] = []
                            else:
                                print(f"Screenshot extraction failed: {ss_result.error_message}")
                    else:
                        anime_data["screenshots"] = []

                # Remove the temporary screenshots_page_url field
                if "screenshots_page_url" in anime_data:
                    del anime_data["screenshots_page_url"]

                # Define relations schema and config
                relations_css_schema = {
                    "baseSelector": "body", # Relations page also has a body tag as base
                    "fields": [
                        {
                            "name": "anime_relations",
                            "selector": "section#relations_anime tbody tr",
                            "type": "nested_list",
                            "fields": [
                                { "name": "type", "selector": "th span", "type": "text" },
                                { "name": "title", "selector": "th a", "type": "text" },
                                { "name": "url", "selector": "th a", "type": "attribute", "attribute": "href" },
                                { "name": "details", "selector": "td[data-title='Type / Episodes / Year']", "type": "text" },
                                { "name": "genres", "selector": "td[data-title='Main genres']", "type": "text" },
                                { "name": "rating", "selector": "td.rating div.star0", "type": "attribute", "attribute": "title" },
                                { "name": "tooltip_html", "selector": "th[scope='row']", "type": "attribute", "attribute": "data-tooltip" }
                            ]
                        },
                        {
                            "name": "manga_relations",
                            "selector": "section#relations_manga tbody tr",
                            "type": "nested_list",
                            "fields": [
                                { "name": "type", "selector": "th span", "type": "text" },
                                { "name": "title", "selector": "th a", "type": "text" },
                                { "name": "url", "selector": "th a", "type": "attribute", "attribute": "href" },
                                { "name": "details", "selector": "td[data-title='Type / Episodes / Year']", "type": "text" },
                                { "name": "genres", "selector": "td[data-title='Main genres']", "type": "text" },
                                { "name": "rating", "selector": "td.rating div.star0", "type": "attribute", "attribute": "title" },
                                { "name": "tooltip_html", "selector": "th[scope='row']", "type": "attribute", "attribute": "data-tooltip" }
                            ]
                        }
                    ]
                }
                relations_extraction_strategy = JsonCssExtractionStrategy(relations_css_schema)
                relations_config = CrawlerRunConfig(extraction_strategy=relations_extraction_strategy)

                # Extract relations from the linked page
                if "relations_page_url" in anime_data and anime_data["relations_page_url"]:
                    relative_url = anime_data["relations_page_url"]
                    # Ensure the relations URL includes "?show=overall"
                    if "?show=overall" not in relative_url:
                        relative_url += "?show=overall"
                    full_relations_url = f"https://www.anisearch.com/{relative_url}"
                    
                    print(f"Fetching relations from: {full_relations_url}")
                    relations_results: RunManyReturn = await crawler.arun(full_relations_url, config=relations_config)

                    if relations_results:
                        for rel_result in relations_results:
                            rel_result = cast(CrawlResult, rel_result)
                            if rel_result.success and rel_result.extracted_content:
                                rel_data = json.loads(rel_result.extracted_content)
                                
                                if rel_data and "anime_relations" in rel_data[0]:
                                    # Process tooltip HTML for anime relations
                                    for relation in rel_data[0]["anime_relations"]:
                                        if "tooltip_html" in relation and relation["tooltip_html"]:
                                            unescaped_html = html.unescape(relation["tooltip_html"])
                                            img_match = re.search(r'<img src="([^"]+)"' , unescaped_html)
                                            if img_match:
                                                relation["image"] = img_match.group(1) # Set image to processed URL
                                            # Remove tooltip_html as it's replaced by image
                                            del relation["tooltip_html"]
                                    anime_data["anime_relations"] = rel_data[0]["anime_relations"]
                                else:
                                    anime_data["anime_relations"] = []
                                
                                if rel_data and "manga_relations" in rel_data[0]:
                                    # Process tooltip HTML for manga relations
                                    for relation in rel_data[0]["manga_relations"]:
                                        if "tooltip_html" in relation and relation["tooltip_html"]:
                                            unescaped_html = html.unescape(relation["tooltip_html"])
                                            img_match = re.search(r'<img src="([^"]+)"' , unescaped_html)
                                            if img_match:
                                                relation["image"] = img_match.group(1) # Set image to processed URL
                                            # Remove tooltip_html as it's replaced by image
                                            del relation["tooltip_html"]
                                    anime_data["manga_relations"] = rel_data[0]["manga_relations"]
                                else:
                                    anime_data["manga_relations"] = []
                            else:
                                print(f"Relation extraction failed: {rel_result.error_message}")
                    else:
                        anime_data["anime_relations"] = []
                        anime_data["manga_relations"] = []

                # Remove the temporary relations_page_url field
                if "relations_page_url" in anime_data:
                    del anime_data["relations_page_url"]

                output_path = "/home/dani/code/anime-vector-service/anisearch_anime.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(anime_data, f, ensure_ascii=False, indent=2)
                print(f"Data written to {output_path}")
            else:
                print(f"Extraction failed: {result.error_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl anime data from an anisearch.com URL.")
    parser.add_argument("url", type=str, help="The anisearch.com URL for the anime page.")
    args = parser.parse_args()
    asyncio.run(fetch_anisearch_anime(args.url))
