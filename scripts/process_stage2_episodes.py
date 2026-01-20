#!/usr/bin/env python3
"""
Process all episodes from episodes_detailed.json and convert to Stage 2 schema format.
Following the prompt template exactly for all 1,144 episodes.
Includes proper timezone conversion from JST (+09:00) to UTC (Z format).
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Project root for resolving paths (works from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert_jst_to_utc(jst_datetime_str):
    """
    Convert JST datetime string to UTC with Z format.

    Example: "1999-10-20T00:00:00+09:00" -> "1999-10-19T15:00:00Z"
    """
    if not jst_datetime_str:
        return None

    try:
        # Parse the JST datetime
        dt = datetime.fromisoformat(jst_datetime_str)

        # Convert to UTC
        utc_dt = dt.astimezone(UTC)

        # Format with Z suffix (same as existing database)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Error converting datetime {jst_datetime_str}: {e}")
        return jst_datetime_str


def load_kitsu_episode_data(temp_dir: str):
    """Load Kitsu episode data and create episode number mappings."""
    try:
        with open(f"{temp_dir}/kitsu.json") as f:
            kitsu_data = json.load(f)

        # Extract anime slug for constructing episode URLs
        anime_slug = kitsu_data.get("anime", {}).get("attributes", {}).get("slug", "")

        kitsu_episodes = kitsu_data.get("episodes", [])

        # Create mappings by episode number
        kitsu_thumbnails = {}
        kitsu_descriptions = {}
        kitsu_synopses = {}
        kitsu_season_numbers = {}
        kitsu_episode_urls = {}

        for episode in kitsu_episodes:
            attrs = episode.get("attributes", {})
            ep_number = attrs.get("number")

            if ep_number:
                # Extract thumbnail URL
                thumbnail = attrs.get("thumbnail", {})
                if thumbnail and thumbnail.get("original"):
                    kitsu_thumbnails[ep_number] = thumbnail["original"]

                # Extract description
                description = attrs.get("description")
                if description and description.strip():
                    kitsu_descriptions[ep_number] = description.strip()

                # Extract synopsis
                synopsis = attrs.get("synopsis")
                if synopsis and synopsis.strip():
                    kitsu_synopses[ep_number] = synopsis.strip()

                # Extract season number
                season_number = attrs.get("seasonNumber")
                if season_number is not None:
                    kitsu_season_numbers[ep_number] = season_number

                # Construct Kitsu episode URL
                if anime_slug:
                    kitsu_episode_urls[ep_number] = (
                        f"https://kitsu.app/anime/{anime_slug}/episodes/{ep_number}"
                    )

        # Also create title mappings
        kitsu_titles = {}
        kitsu_titles_japanese = {}
        kitsu_titles_romaji = {}

        for episode in kitsu_episodes:
            attrs = episode.get("attributes", {})
            ep_number = attrs.get("number")

            if ep_number:
                titles = attrs.get("titles", {})

                # Extract English title (try 'en' first, then 'en_us')
                if titles.get("en"):
                    kitsu_titles[ep_number] = titles["en"]
                elif titles.get("en_us"):
                    kitsu_titles[ep_number] = titles["en_us"]

                # Extract Japanese title (ja_jp)
                if titles.get("ja_jp"):
                    kitsu_titles_japanese[ep_number] = titles["ja_jp"]

                # Extract Romanji title (en_jp)
                if titles.get("en_jp"):
                    kitsu_titles_romaji[ep_number] = titles["en_jp"]

        print(
            f"Loaded Kitsu data: {len(kitsu_thumbnails)} thumbnails, {len(kitsu_descriptions)} descriptions, {len(kitsu_synopses)} synopses, {len(kitsu_titles)} titles, {len(kitsu_titles_japanese)} ja_jp titles, {len(kitsu_titles_romaji)} en_jp titles, {len(kitsu_season_numbers)} season numbers, {len(kitsu_episode_urls)} episode URLs"
        )
        return (
            kitsu_thumbnails,
            kitsu_descriptions,
            kitsu_synopses,
            kitsu_titles,
            kitsu_titles_japanese,
            kitsu_titles_romaji,
            kitsu_season_numbers,
            kitsu_episode_urls,
        )

    except FileNotFoundError:
        print("Kitsu data not found, proceeding without Kitsu enhancement")
        return {}, {}, {}, {}, {}, {}, {}, {}
    except Exception as e:
        print(f"Error loading Kitsu data: {e}")
        return {}, {}, {}, {}, {}, {}, {}, {}


def load_anisearch_episode_data(temp_dir: str):
    """Load AniSearch episode data and create episode number mappings."""
    try:
        with open(f"{temp_dir}/anisearch.json") as f:
            anisearch_data = json.load(f)

        anisearch_episodes = anisearch_data.get("episodes", [])

        # Create mappings by episode number
        anisearch_titles = {}

        for episode in anisearch_episodes:
            ep_number = episode.get("episodeNumber")
            title = episode.get("title")

            if ep_number and title:
                anisearch_titles[ep_number] = title

        print(f"Loaded AniSearch data: {len(anisearch_titles)} titles")
        return anisearch_titles

    except FileNotFoundError:
        print("AniSearch data not found, proceeding without AniSearch enhancement")
        return {}
    except Exception as e:
        print(f"Error loading AniSearch data: {e}")
        return {}


def process_all_episodes(temp_dir: str):
    # Read the detailed episodes data
    with open(f"{temp_dir}/episodes_detailed.json") as f:
        episodes_data = json.load(f)

    # Load Kitsu episode data for enhancement
    (
        kitsu_thumbnails,
        kitsu_descriptions,
        kitsu_synopses,
        kitsu_titles,
        kitsu_titles_japanese,
        kitsu_titles_romaji,
        kitsu_season_numbers,
        kitsu_episode_urls,
    ) = load_kitsu_episode_data(temp_dir)

    # Load AniSearch episode data for enhancement
    anisearch_titles = load_anisearch_episode_data(temp_dir)

    print(f"Processing {len(episodes_data)} episodes...")

    # Convert each episode to the Stage 2 schema format
    episodes = []

    for episode in episodes_data:
        ep_number = episode.get("episode_number")

        # Get Kitsu enhancements for this episode (match by episode number)
        kitsu_thumbnail = kitsu_thumbnails.get(ep_number)
        kitsu_description = kitsu_descriptions.get(ep_number)
        kitsu_synopsis = kitsu_synopses.get(ep_number)
        kitsu_title = kitsu_titles.get(ep_number)
        kitsu_title_japanese = kitsu_titles_japanese.get(ep_number)
        kitsu_title_romaji = kitsu_titles_romaji.get(ep_number)
        kitsu_season_number = kitsu_season_numbers.get(ep_number)
        kitsu_episode_url = kitsu_episode_urls.get(ep_number)

        # Get AniSearch enhancements for this episode (match by episode number)
        anisearch_title = anisearch_titles.get(ep_number)

        # Build thumbnails array
        thumbnails = []
        if kitsu_thumbnail:
            thumbnails.append(kitsu_thumbnail)

        # Build episode_pages object
        episode_pages = {}
        if episode.get("url"):
            episode_pages["mal"] = episode.get("url")
        if kitsu_episode_url:
            episode_pages["kitsu"] = kitsu_episode_url

        # Convert according to Stage 2 prompt template schema with timezone conversion
        processed_episode = {
            # SCALAR FIELDS (alphabetical)
            "aired": convert_jst_to_utc(episode.get("aired")),
            "description": kitsu_description,  # From Kitsu
            "duration": episode.get("duration"),
            "episode_number": episode.get("episode_number"),
            "filler": episode.get("filler", False),
            "recap": episode.get("recap", False),
            "score": episode.get("score"),
            "season_number": kitsu_season_number,  # From Kitsu (Jikan doesn't provide this)
            "synopsis": episode.get("synopsis")
            or kitsu_synopsis,  # Jikan primary, Kitsu fallback
            "title": episode.get("title")
            or kitsu_title
            or anisearch_title,  # Jikan → Kitsu → AniSearch
            "title_japanese": episode.get("title_japanese")
            or kitsu_title_japanese,  # Jikan primary, Kitsu fallback
            "title_romaji": episode.get("title_romaji")
            or kitsu_title_romaji,  # Jikan primary, Kitsu fallback
            # ARRAY FIELDS (alphabetical)
            "thumbnails": thumbnails,
            # OBJECT/DICT FIELDS (alphabetical)
            "episode_pages": episode_pages,
            "streaming": {},  # No streaming data from Jikan
        }

        episodes.append(processed_episode)

    # Create the final output structure
    output = {"episodes": episodes}

    # Write to stage2_episodes.json
    output_path = f"{temp_dir}/stage2_episodes.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(episodes)} episodes to {output_path}")
    print(
        f"File size: {len(json.dumps(output, indent=2, ensure_ascii=False))} characters"
    )

    # Show some examples of timezone conversion
    print("\nTimezone conversion examples:")
    for i in range(min(3, len(episodes_data))):
        original = episodes_data[i].get("aired")
        converted = episodes[i]["aired"]
        print(f"  Episode {i + 1}: {original} -> {converted}")


def auto_detect_temp_dir():
    """Auto-detect the temp directory based on available directories."""
    temp_base = "temp"
    if not os.path.exists(temp_base):
        print(f"Error: {temp_base} directory not found")
        sys.exit(1)

    # Look for directories in temp/
    temp_dirs = [
        d for d in os.listdir(temp_base) if os.path.isdir(os.path.join(temp_base, d))
    ]

    if not temp_dirs:
        print(f"Error: No anime directories found in {temp_base}/")
        sys.exit(1)

    if len(temp_dirs) == 1:
        detected_dir = os.path.join(temp_base, temp_dirs[0])
        print(f"Auto-detected temp directory: {detected_dir}")
        return detected_dir
    else:
        print(f"Multiple temp directories found: {temp_dirs}")
        print("Please specify which one to use with --temp-dir argument")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Stage 2: Episode processing with multi-source integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_stage2_episodes.py One_agent1                         # Process One_agent1 directory
  python process_stage2_episodes.py Dandadan_agent1                    # Process Dandadan_agent1 directory
  python process_stage2_episodes.py One_agent1 --temp-dir custom_temp # Use custom temp directory
        """,
    )
    parser.add_argument(
        "agent_id",
        nargs="?",
        help="Agent directory name to process (e.g., One_agent1, Dandadan_agent1)",
    )
    parser.add_argument(
        "--temp-dir", default="temp", help="Temporary directory path (default: temp)"
    )

    args = parser.parse_args()

    # Determine temp_dir and construct full path (relative to project root)
    if args.agent_id:
        # New pattern: agent_id + temp_dir
        temp_base = Path(args.temp_dir)
        if not temp_base.is_absolute():
            temp_base = PROJECT_ROOT / temp_base
        temp_dir = str(temp_base / args.agent_id)
        if not os.path.exists(temp_dir):
            print(f"Error: Directory '{temp_dir}' does not exist")
            sys.exit(1)
    else:
        # Fallback: auto-detect
        temp_dir = auto_detect_temp_dir()

    # Check if episodes_detailed.json exists before processing
    episodes_file = f"{temp_dir}/episodes_detailed.json"
    if not os.path.exists(episodes_file):
        print(f"Error: Required file not found: {episodes_file}")
        print("Please ensure the API fetcher has created episodes_detailed.json")
        sys.exit(1)

    process_all_episodes(temp_dir)
