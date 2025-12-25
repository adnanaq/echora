#!/usr/bin/env python3
"""
Multi-source relationship analysis according to Stage 3 prompt requirements.
Processes Jikan, AnimePlanet, AnimSchedule, AniList, AniDB, and offline URLs data with intelligent deduplication.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Project root for resolving paths (works from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def clean_animeplanet_title(title: str) -> str:
    """
    Clean AnimePlanet titles by removing date/episode information.
    Examples:
      "Series Name Season 2 2024-01-01 - 2024-03-31 TV: 12 ep" → "Series Name Season 2"
      "Movie Title 2023-12-01 - 2023-12-01 Movie" → "Movie Title"
    """
    # Remove date patterns and episode counts
    patterns = [
        r"\s+\d{4}-\d{2}-\d{2}\s+-\s+\d{4}-\d{2}-\d{2}.*$",  # Date ranges
        r"\s+\w+:\s+\d+\s+ep$",  # Episode counts like "TV: 12 ep"
        r"\s+\w+$",  # Trailing type like "Movie", "OVA"
    ]

    cleaned = title
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    return cleaned.strip()


def map_animeplanet_relationship(
    relation_type: str, relation_subtype: str | None = None
) -> str:
    """
    Map AnimePlanet relationship types to standardized types.
    Use subtype when available, otherwise map relation_type.
    """
    # Prefer subtype when available
    if relation_subtype:
        subtype_mapping = {
            "Sequel": "Sequel",
            "Prequel": "Prequel",
            "Alternative Version": "Alternative Version",
            "Side Story": "Side Story",
            "Spin-off": "Spin-off",
            "Movie": "Movie",
            "OVA": "OVA",
            "Special": "Special",
            "Music Video": "Music Video",
            "Live Action": "Live Action",
        }
        if relation_subtype in subtype_mapping:
            return subtype_mapping[relation_subtype]

    # Fallback to generic mapping
    if relation_type == "same_franchise":
        return "Other"

    return relation_type


def map_animeschedule_relationship(category: str) -> str:
    """
    Map AnimSchedule categories to relationship types.
    """
    mapping = {
        "sequels": "Sequel",
        "prequels": "Prequel",
        "sideStories": "Side Story",
        "movies": "Movie",
        "specials": "Special",
        "adaptations": "Adaptation",
        "music": "Music Video",
        "liveAction": "Live Action",
        "other": "Other",
        "alternatives": "Alternative Version",
    }
    return mapping.get(category, "Other")


def route_to_url(route: str) -> str:
    """
    Convert AnimSchedule route to full URL.
    """
    return f"https://animeschedule.net/anime/{route}"


def route_to_title(route: str) -> str:
    """
    Extract title from route using intelligent slug parsing.
    """
    # Convert kebab-case to title case
    title = route.replace("-", " ")
    # Capitalize words
    title = " ".join(word.capitalize() for word in title.split())
    return title


def process_jikan_relations(
    jikan_data: dict[str, Any],
) -> tuple[list[dict], list[dict]]:
    """
    Process Jikan relations data (primary source).
    Returns (related_anime, relations) tuples.
    """
    related_anime = []
    relations = []

    jikan_relations = jikan_data.get("data", {}).get("relations", [])

    for relation_group in jikan_relations:
        relation_type = relation_group.get("relation")
        entries = relation_group.get("entry", [])

        for entry in entries:
            entry_data = {
                "relation_type": relation_type,
                "title": entry.get("name"),
                "url": entry.get("url"),
            }

            # Separate by type: anime vs manga
            if entry.get("type") == "anime":
                related_anime.append(entry_data)
            elif entry.get("type") == "manga":
                relations.append(entry_data)

    return related_anime, relations


def process_animeplanet_relations(animeplanet_data: dict[str, Any]) -> list[dict]:
    """
    Process AnimePlanet relations (co-primary source).
    """
    related_anime = []
    animeplanet_relations = animeplanet_data.get("related_anime", [])

    for relation in animeplanet_relations:
        # Clean title
        raw_title = relation.get("title", "")
        cleaned_title = clean_animeplanet_title(raw_title)

        # Map relationship type
        relation_type = relation.get("relation_type", "")
        relation_subtype = relation.get("relation_subtype")
        mapped_type = map_animeplanet_relationship(relation_type, relation_subtype)

        entry_data = {
            "relation_type": mapped_type,
            "title": cleaned_title,
            "url": relation.get("url"),
        }
        related_anime.append(entry_data)

    return related_anime


def process_animeschedule_relations(animeschedule_data: dict[str, Any]) -> list[dict]:
    """
    Process AnimSchedule relations (supplementary).
    """
    related_anime = []
    relations_data = animeschedule_data.get("relations", {})

    for category, routes in relations_data.items():
        mapped_type = map_animeschedule_relationship(category)

        for route in routes:
            entry_data = {
                "relation_type": mapped_type,
                "title": route_to_title(route),
                "url": route_to_url(route),
            }
            related_anime.append(entry_data)

    return related_anime


def map_anilist_relationship(relation_type: str) -> str:
    """
    Map AniList relationship types to standardized types.
    """
    mapping = {
        "SIDE_STORY": "Side Story",
        "SUMMARY": "Summary",
        "SEQUEL": "Sequel",
        "PREQUEL": "Prequel",
        "ADAPTATION": "Adaptation",
        "SPIN_OFF": "Spin-off",
        "ALTERNATIVE": "Alternative Version",
        "CHARACTER": "Character",
        "PARENT": "Parent Story",
        "FULL_STORY": "Full Story",
        "OTHER": "Other",
    }
    return mapping.get(relation_type, "Other")


def process_anilist_relations(
    anilist_data: dict[str, Any],
) -> tuple[list[dict], list[dict]]:
    """
    Process AniList relations data.
    Returns (related_anime, relations) tuples.
    """
    related_anime = []
    relations = []

    anilist_relations = anilist_data.get("relations", {}).get("edges", [])

    for edge in anilist_relations:
        node = edge.get("node", {})
        relation_type = edge.get("relationType", "")

        # Extract title - prefer romaji, fallback to english
        title_obj = node.get("title", {})
        title = title_obj.get("romaji") or title_obj.get("english") or "Unknown Title"

        # Create AniList URL from ID
        node_id = node.get("id")
        url = f"https://anilist.co/anime/{node_id}" if node_id else ""

        entry_data = {
            "relation_type": map_anilist_relationship(relation_type),
            "title": title,
            "url": url,
        }

        # AniList only has anime entries in relations
        related_anime.append(entry_data)

    return related_anime, relations


def process_anidb_relations(anidb_data: dict[str, Any]) -> list[dict]:
    """
    Process AniDB relations data (if available).
    Note: AniDB doesn't typically have explicit relations in the API response.
    """
    related_anime = []

    # AniDB data structure doesn't typically include relations
    # This is a placeholder for consistency

    return related_anime


def extract_current_anime_related_urls(current_anime_file: str) -> list[str]:
    """
    Extract related anime URLs from current anime being processed.
    Following enrichment_instructions.md Step 3 requirements.
    """
    try:
        with open(current_anime_file) as f:
            current_anime = json.load(f)

        # Get related anime URLs from the current anime being processed
        related_urls = current_anime.get("relatedAnime", [])

        print(f"  - Current anime: {current_anime.get('title', 'Unknown')}")
        print(f"  - Related anime URLs from current anime: {len(related_urls)}")

        return related_urls
    except Exception as e:
        print(f"Error loading current anime file {current_anime_file}: {e}")
        return []


def process_offline_urls(
    offline_urls: list[str], existing_urls: set[str], existing_mal_base_urls: set[str]
) -> list[dict]:
    """
    Process offline URLs data according to Stage 3 prompt requirements.

    Following the prompt:
    1. Check URL deduplication first
    2. Process ALL remaining URLs (never skip due to title extraction failure)
    3. Use intelligent title extraction with WebFetch/helper functions simulation
    4. Add entries even if title extraction fails

    Args:
        offline_urls: List of URLs from offline database
        existing_urls: Set of URLs already present in Jikan/AnimePlanet relations
    """
    related_anime = []
    skipped_count = 0
    processed_count = 0

    for url in offline_urls:
        if not url:
            continue

        # STEP 1: Check if URL already exists in non-offline relations → skip if found
        normalized_url = url.lower().replace("www.", "").rstrip("/")
        is_duplicate = normalized_url in existing_urls

        # Additional check for MAL base URL duplication
        if not is_duplicate and "myanimelist.net/anime/" in url.lower():
            import re

            match = re.search(r"(myanimelist\.net/anime/\d+)", url.lower())
            if match:
                mal_base_url = f"https://{match.group(1)}"
                if mal_base_url in existing_mal_base_urls:
                    is_duplicate = True

        if is_duplicate:
            skipped_count += 1
            continue

        # STEP 2: Process ALL remaining URLs - never skip due to title extraction failure
        title = extract_title_from_url(url)
        relation_type = infer_relationship_from_url(url)

        # If title extraction failed, use URL-based fallback but still include the entry
        if not title or title == "Unknown":
            title = create_fallback_title_from_url(url)

        # ALWAYS create entry - following prompt requirement to process EVERY URL
        entry_data = {"relation_type": relation_type, "title": title, "url": url}
        related_anime.append(entry_data)
        processed_count += 1

    print(
        f"  - Offline URL processing: {skipped_count} URLs skipped (already in Jikan/AnimePlanet)"
    )
    print(
        f"  - Offline URL processing: {processed_count} URLs processed (all remaining URLs included)"
    )
    return related_anime


def create_fallback_title_from_url(url: str) -> str:
    """
    Create a meaningful fallback title when URL parsing fails.
    Still better than filtering out the relationship entirely.
    """
    if not url:
        return "Related Anime"

    # Extract domain for context
    if "anidb.net" in url:
        return "Related Anime (AniDB)"
    elif "anisearch.com" in url:
        return "Related Anime (AniSearch)"
    elif "animenewsnetwork.com" in url:
        return "Related Anime (ANN)"
    elif "simkl.com" in url:
        return "Related Anime (Simkl)"
    elif "notify.moe" in url:
        return "Related Anime (Notify.moe)"
    elif "livechart.me" in url:
        return "Related Anime (LiveChart)"
    else:
        return "Related Anime"


def extract_title_from_url(url: str) -> str:
    """
    Extract title from URL (simplified version).
    """
    if not url:
        return "Unknown"

    # Extract from different platforms
    if "myanimelist.net" in url:
        # Extract from MAL URL pattern
        parts = url.split("/")
        if len(parts) > 4:
            title_part = parts[4]
            return title_part.replace("_", " ").replace("-", " ").title()

    elif "anime-planet.com" in url:
        # Extract from AnimePlanet URL pattern
        parts = url.split("/")
        if len(parts) > 4:
            title_part = parts[4]
            return title_part.replace("-", " ").title()

    elif "anilist.co" in url:
        # AniList URLs don't contain titles, return generic
        return "Related Anime"

    elif "kitsu.app" in url or "kitsu.io" in url:
        # Kitsu URLs don't contain titles, return generic
        return "Related Anime"

    return "Unknown"


def infer_relationship_from_url(url: str) -> str:
    """
    Infer relationship type from URL context (simplified version).
    """
    url_lower = url.lower()

    if "movie" in url_lower:
        return "Movie"
    elif "ova" in url_lower:
        return "OVA"
    elif "special" in url_lower:
        return "Special"
    elif "season" in url_lower:
        return "Sequel"
    else:
        return "Other"


def deduplicate_by_url(all_relations: list[dict]) -> list[dict]:
    """
    Deduplicate relations using URL-based strategy with priority hierarchy.
    Priority: Jikan > AnimePlanet > AnimSchedule > Offline URLs
    """
    url_groups = {}

    # Group by URL (normalize URLs first)
    for relation in all_relations:
        url = relation.get("url", "")
        # Normalize URL (remove www, trailing slashes, etc.)
        normalized_url = url.lower().replace("www.", "").rstrip("/")

        if normalized_url not in url_groups:
            url_groups[normalized_url] = []
        url_groups[normalized_url].append(relation)

    # Select best entry per URL group using priority
    deduplicated = []

    for url, entries in url_groups.items():
        if len(entries) == 1:
            deduplicated.append(entries[0])
        else:
            # Multiple entries for same URL - apply priority hierarchy
            best_entry = entries[0]

            # Source priority: Jikan > AnimePlanet > AnimSchedule
            for entry in entries:
                url = entry.get("url", "")
                if "myanimelist.net" in url:  # Jikan source
                    best_entry = entry
                    break
                elif "anime-planet.com" in url:  # AnimePlanet source
                    best_entry = entry
                elif (
                    "animeschedule.net" in url
                    and "anime-planet.com" not in best_entry.get("url", "")
                ):
                    best_entry = entry

            deduplicated.append(best_entry)

    return deduplicated


def process_all_relationships(current_anime_file: str, temp_dir: str):
    """
    Main processing function following Stage 3 prompt requirements.
    Processes all 6 sources: Jikan, AnimePlanet, AnimSchedule, AniList, AniDB, and Offline URLs.

    Args:
        current_anime_file: Path to current_anime_N.json file
        temp_dir: Path to temp directory with data files (e.g., temp/One/)
    """
    # Load all data sources from the specified temp directory
    with open(f"{temp_dir}/jikan.json") as f:
        jikan_data = json.load(f)

    with open(f"{temp_dir}/anime_planet.json") as f:
        animeplanet_data = json.load(f)

    with open(f"{temp_dir}/animeschedule.json") as f:
        animeschedule_data = json.load(f)

    with open(f"{temp_dir}/anilist.json") as f:
        anilist_data = json.load(f)

    with open(f"{temp_dir}/anidb.json") as f:
        anidb_data = json.load(f)

    print("Processing relationships from 6 sources...")

    # Step 1: Process Jikan relations (primary source)
    jikan_anime, jikan_manga = process_jikan_relations(jikan_data)
    print(
        f"Jikan: {len(jikan_anime)} anime relations, {len(jikan_manga)} manga relations"
    )

    # Step 2: Process AnimePlanet relations (co-primary source)
    animeplanet_anime = process_animeplanet_relations(animeplanet_data)
    print(f"AnimePlanet: {len(animeplanet_anime)} anime relations")

    # Step 3: Process AnimSchedule relations (supplementary)
    animeschedule_anime = process_animeschedule_relations(animeschedule_data)
    print(f"AnimSchedule: {len(animeschedule_anime)} anime relations")

    # Step 4: Process AniList relations
    anilist_anime, anilist_manga = process_anilist_relations(anilist_data)
    print(
        f"AniList: {len(anilist_anime)} anime relations, {len(anilist_manga)} manga relations"
    )

    # Step 5: Process AniDB relations
    anidb_anime = process_anidb_relations(anidb_data)
    print(f"AniDB: {len(anidb_anime)} anime relations")

    # Step 6: Combine and deduplicate all NON-OFFLINE sources first
    non_offline_relations = (
        jikan_anime
        + animeplanet_anime
        + animeschedule_anime
        + anilist_anime
        + anidb_anime
    )
    print(f"Non-offline sources total: {len(non_offline_relations)} anime relations")

    # Step 7: Deduplicate non-offline sources
    deduplicated_non_offline = deduplicate_by_url(non_offline_relations)
    print(
        f"After non-offline deduplication: {len(deduplicated_non_offline)} anime relations"
    )

    # Step 8: Create set of existing URLs and MAL base URLs from deduplicated non-offline relations
    existing_urls = set()
    existing_mal_base_urls = set()

    for relation in deduplicated_non_offline:
        if relation.get("url"):
            url = relation["url"]
            normalized_url = url.lower().replace("www.", "").rstrip("/")
            existing_urls.add(normalized_url)

            # Create MAL base URL if it's a MyAnimeList URL
            if "myanimelist.net/anime/" in url.lower():
                import re

                match = re.search(r"(myanimelist\.net/anime/\d+)", url.lower())
                if match:
                    mal_base_url = f"https://{match.group(1)}"
                    existing_mal_base_urls.add(mal_base_url)

    # Step 9: Process Offline URLs (check against FINAL deduplicated list)
    offline_urls = extract_current_anime_related_urls(current_anime_file)

    # VERIFICATION: Count what will be skipped vs processed BEFORE processing
    verification_skipped = 0
    verification_processed = 0

    for url in offline_urls:
        if not url:
            continue

        # Check standard URL duplication
        normalized_url = url.lower().replace("www.", "").rstrip("/")
        is_duplicate = normalized_url in existing_urls

        # Additional check for MAL base URL duplication
        if not is_duplicate and "myanimelist.net/anime/" in url.lower():
            import re

            match = re.search(r"(myanimelist\.net/anime/\d+)", url.lower())
            if match:
                mal_base_url = f"https://{match.group(1)}"
                if mal_base_url in existing_mal_base_urls:
                    is_duplicate = True

        if is_duplicate:
            verification_skipped += 1
        else:
            verification_processed += 1

    print(f"  - VERIFICATION: {verification_skipped} URLs will be skipped (duplicates)")
    print(f"  - VERIFICATION: {verification_processed} URLs will be processed (new)")

    # Now actually process the offline URLs
    offline_anime = process_offline_urls(
        offline_urls, existing_urls, existing_mal_base_urls
    )
    print(
        f"Offline URLs: {len(offline_urls)} URLs extracted, {len(offline_anime)} anime relations processed"
    )

    # VERIFICATION: Confirm the counts match
    if len(offline_anime) == verification_processed:
        print(
            f"  ✅ VERIFICATION PASSED: Processed count matches ({len(offline_anime)})"
        )
    else:
        print(
            f"  ❌ VERIFICATION FAILED: Expected {verification_processed}, got {len(offline_anime)}"
        )

    actual_skipped = len(offline_urls) - len(offline_anime)
    if actual_skipped == verification_skipped:
        print(f"  ✅ VERIFICATION PASSED: Skipped count matches ({actual_skipped})")
    else:
        print(
            f"  ❌ VERIFICATION FAILED: Expected {verification_skipped} skipped, got {actual_skipped}"
        )

    # Step 10: Combine deduplicated non-offline with processed offline
    all_anime_relations = deduplicated_non_offline + offline_anime
    print(
        f"Final total: {len(all_anime_relations)} anime relations (no further deduplication needed)"
    )

    # Combine manga relations from sources that have them
    all_manga_relations = jikan_manga + anilist_manga
    print(f"Total manga relations: {len(all_manga_relations)}")

    # Create final output
    output = {"related_anime": all_anime_relations, "relations": all_manga_relations}

    # Write output
    output_file = f"{temp_dir}/stage3_relationships.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Stage 3 processing complete:")
    print(f"  - Related anime: {len(all_anime_relations)} entries")
    print(f"  - Relations (manga): {len(all_manga_relations)} entries")
    print(
        "  - Sources processed: Jikan, AnimePlanet, AnimSchedule, AniList, AniDB, Offline URLs"
    )
    print(f"  - File saved: {output_file}")


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


def auto_detect_current_anime_file(temp_dir: str):
    """Auto-detect current anime file following consistent directory structure."""
    # First, check if current_anime.json exists in the provided temp_dir
    current_anime_in_dir = os.path.join(temp_dir, "current_anime.json")
    if os.path.exists(current_anime_in_dir):
        print(f"Found current anime file: {current_anime_in_dir}")
        return current_anime_in_dir

    # Fallback: Look for current_anime.json files in agent directories under temp/
    temp_base = "temp"
    if os.path.exists(temp_base):
        agent_dirs = []
        for item in os.listdir(temp_base):
            item_path = os.path.join(temp_base, item)
            if os.path.isdir(item_path):
                current_anime_file = os.path.join(item_path, "current_anime.json")
                if os.path.exists(current_anime_file):
                    agent_dirs.append((item, current_anime_file))

        if agent_dirs:
            if len(agent_dirs) == 1:
                dir_name, detected_file = agent_dirs[0]
                print(f"Auto-detected current anime file: {detected_file}")
                return detected_file
            else:
                print(
                    f"Multiple agent directories with current_anime.json found: {[d[0] for d in agent_dirs]}"
                )
                print("Please specify which one to use with --current-anime argument")
                sys.exit(1)

    print(
        f"Error: No current anime file found. Looked in {temp_dir}/ and agent directories in temp/"
    )
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Stage 3: Multi-source relationship analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_stage3_relationships.py One_agent1                         # Process One_agent1 directory
  python process_stage3_relationships.py Dandadan_agent1                    # Process Dandadan_agent1 directory
  python process_stage3_relationships.py One_agent1 --temp-dir custom_temp # Use custom temp directory
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
    parser.add_argument(
        "--current-anime",
        type=str,
        help="Override current anime JSON file path (optional, for backward compatibility)",
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
    elif args.current_anime:
        # Legacy pattern: derive from file path
        temp_dir = os.path.dirname(args.current_anime)
        print(f"Derived temp directory from file path: {temp_dir}")
    else:
        # Fallback: auto-detect
        temp_dir = auto_detect_temp_dir()

    # Determine current_anime_file
    if args.current_anime:
        current_anime_file = args.current_anime
        if not os.path.exists(current_anime_file):
            print(
                f"Error: Specified current anime file '{current_anime_file}' does not exist"
            )
            sys.exit(1)
    else:
        current_anime_file = auto_detect_current_anime_file(temp_dir)

    print("Processing with:")
    print(f"  Current anime file: {current_anime_file}")
    print(f"  Temp directory: {temp_dir}")

    process_all_relationships(current_anime_file, temp_dir)
