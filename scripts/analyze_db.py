"""This module provides functionality to analyze differences between two anime database JSON files."""

import argparse
import collections
import json
import re
import sys
from typing import Any

# Pre-compile regex for performance
MAL_REGEX = re.compile(r"myanimelist\.net/anime/(\d+)")
ANILIST_REGEX = re.compile(r"anilist\.co/anime/(\d+)")
KITSU_REGEX = re.compile(r"kitsu\.(?:io|app)/anime/(\d+)")
ANIDB_REGEX = re.compile(r"anidb\.net/anime/(\d+)")
ANIMEPLANET_REGEX = re.compile(r"anime-planet\.com/anime/([^/]+)")


def get_stable_id(entry: dict[str, Any]) -> str:
    """Extracts a stable ID from the entry's sources or falls back to title.

    Priority: MAL > AniList > Kitsu > AniDB > AnimePlanet > Title.

    Args:
        entry: A dictionary representing an anime entry.

    Returns:
        A string representing a stable, unique-ish ID for the anime.
    """
    sources = entry.get("sources", [])

    # Check for MAL
    for source in sources:
        match = MAL_REGEX.search(source)
        if match:
            return f"mal:{match.group(1)}"

    # Check for AniList
    for source in sources:
        match = ANILIST_REGEX.search(source)
        if match:
            return f"anilist:{match.group(1)}"

    # Check for Kitsu
    for source in sources:
        match = KITSU_REGEX.search(source)
        if match:
            return f"kitsu:{match.group(1)}"

    # Check for AniDB
    for source in sources:
        match = ANIDB_REGEX.search(source)
        if match:
            return f"anidb:{match.group(1)}"

    # Check for AnimePlanet
    for source in sources:
        match = ANIMEPLANET_REGEX.search(source)
        if match:
            return f"animeplanet:{match.group(1)}"

    # Fallback to title + type (to avoid collisions on remakes with same name if strictly title)
    title = entry.get("title", "unknown").lower()
    anime_type = entry.get("type", "unknown").lower()
    return f"fallback:{title}|{anime_type}"


def load_data(filepath: str) -> dict[str, dict[str, Any]]:
    """Loads anime data from a JSON file and maps it by stable ID.

    Args:
        filepath: The path to the JSON file to load.

    Returns:
        A dictionary mapping stable IDs to their respective anime entry data.
    """
    print(f"Loading {filepath}...")
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            # Support both raw list and {'data': [...]} format
            entries = data.get("data", data) if isinstance(data, dict) else data

            id_map = {}
            duplicates = 0
            for entry in entries:
                uid = get_stable_id(entry)
                if uid in id_map:
                    duplicates += 1
                id_map[uid] = entry

            if duplicates > 0:
                print(
                    f"Warning: {duplicates} duplicate IDs found in {filepath} (mostly fallback IDs)."
                )

            return id_map
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def diff_entries(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Finds field-level differences between two anime entries.

    Special handling is included for list fields where order might not matter.

    Args:
        old: The original anime entry.
        new: The new anime entry to compare against.

    Returns:
        A dictionary mapping changed field names to their old and new values.
    """
    diffs = {}
    all_keys = set(old.keys()) | set(new.keys())

    for key in all_keys:
        val_old = old.get(key)
        val_new = new.get(key)

        # Simple comparison
        if val_old != val_new:
            # Special handling for lists where order might not matter
            if isinstance(val_old, list) and isinstance(val_new, list):
                if sorted(str(x) for x in val_old) == sorted(str(x) for x in val_new):
                    continue

            diffs[key] = {"old": val_old, "new": val_new}

    return diffs


def analyze_added_years(added_entries: list[dict[str, Any]], output_file: Any) -> None:
    """Analyzes and reports the release years and eras of newly added entries.

    Output is formatted into a three-column layout for readability.

    Args:
        added_entries: A list of newly added anime entries.
        output_file: A file object where the report will be written.
    """
    year_counts = collections.Counter()

    for entry in added_entries:
        year = entry.get("animeSeason", {}).get("year")
        if year is None:
            year = "Unknown"
        year_counts[year] += 1

    print("\nRelease Years of Added Entries (Year Order):")
    header = f"{'Year':<10} | {'Count':<6}    {'Year':<10} | {'Count':<6}    {'Year':<10} | {'Count':<6}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    output_file.write(
        "\nRELEASE YEARS OF ADDED ENTRIES\n------------------------------\n"
    )
    output_file.write(f"{header}\n{separator}\n")

    # Sort years: numeric years first, then "Unknown"
    sorted_years = sorted(
        [y for y in year_counts.keys() if y != "Unknown"], key=lambda x: int(x)
    )
    if "Unknown" in year_counts:
        sorted_years.append("Unknown")

    # Split into three columns for side-by-side display
    rows = (len(sorted_years) + 2) // 3
    col1 = sorted_years[:rows]
    col2 = sorted_years[rows : rows * 2]
    col3 = sorted_years[rows * 2 :]

    for i in range(rows):
        y1 = col1[i]
        c1 = year_counts[y1]
        line = f"{str(y1):<10} | {c1:<6}"

        if i < len(col2):
            y2 = col2[i]
            c2 = year_counts[y2]
            line += f"    {str(y2):<10} | {c2:<6}"

        if i < len(col3):
            y3 = col3[i]
            c3 = year_counts[y3]
            line += f"    {str(y3):<10} | {c3:<6}"

        print(line)
        output_file.write(f"{line}\n")

    # Group by Era
    eras = {
        "Pre-2000": 0,
        "2000-2010": 0,
        "2011-2020": 0,
        "2021-2023": 0,
        "2024": 0,
        "2025": 0,
        "Future (2026+)": 0,
        "Unknown": 0,
    }

    for year, count in year_counts.items():
        if year == "Unknown":
            eras["Unknown"] += count
            continue

        try:
            y = int(year)
            if y < 2000:
                eras["Pre-2000"] += count
            elif y <= 2010:
                eras["2000-2010"] += count
            elif y <= 2020:
                eras["2011-2020"] += count
            elif y <= 2023:
                eras["2021-2023"] += count
            elif y == 2024:
                eras["2024"] += count
            elif y == 2025:
                eras["2025"] += count
            else:
                eras["Future (2026+)"] += count
        except (ValueError, TypeError):
            eras["Unknown"] += count

    print("\nSummary by Era:")
    output_file.write("\nSUMMARY BY ERA\n--------------\n")
    for era, count in eras.items():
        print(f"{era}: {count}")
        output_file.write(f"{era}: {count}\n")


def main() -> None:
    """Main execution function to handle arguments and initiate analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze differences between two anime database JSON files."
    )
    parser.add_argument(
        "old_file",
        nargs="?",
        default="data/qdrant_storage/anime-offline-database-minified_old.json",
        help="Path to the old JSON file",
    )
    parser.add_argument(
        "new_file",
        nargs="?",
        default="data/qdrant_storage/anime-offline-database-minified_new.json",
        help="Path to the new JSON file",
    )
    parser.add_argument(
        "--report",
        default="db_analysis_report.txt",
        help="Output path for the analysis report",
    )
    args = parser.parse_args()

    map_old = load_data(args.old_file)
    map_new = load_data(args.new_file)

    keys_old = set(map_old.keys())
    keys_new = set(map_new.keys())

    removed_keys = keys_old - keys_new
    added_keys = keys_new - keys_old
    common_keys = keys_old.intersection(keys_new)

    changed_entries = {}
    field_change_counts = {}
    identical_count = 0

    print("Analyzing changes...")
    for uid in common_keys:
        old_entry = map_old[uid]
        new_entry = map_new[uid]

        # Optimization: string comparison first
        if json.dumps(old_entry, sort_keys=True) == json.dumps(
            new_entry, sort_keys=True
        ):
            identical_count += 1
            continue

        diff = diff_entries(old_entry, new_entry)
        if diff:
            changed_entries[uid] = {"title": new_entry.get("title"), "diff": diff}
            for field in diff:
                field_change_counts[field] = field_change_counts.get(field, 0) + 1
        else:
            identical_count += 1

    # Console Output
    print("\nAnalysis Complete.")
    print(f"Total Old Entries: {len(map_old)}")
    print(f"Total New Entries: {len(map_new)}")
    print("-----------------------------------")
    print(f"Identical Entries: {identical_count}")
    print(f"Removed Entries:   {len(removed_keys)}")
    print(f"Added Entries:     {len(added_keys)}")
    print(f"Changed Entries:   {len(changed_entries)}")
    print("-----------------------------------")

    if field_change_counts:
        print("Field Change Frequency (Top 10):")
        sorted_fields = sorted(
            field_change_counts.items(), key=lambda x: x[1], reverse=True
        )
        for field, count in sorted_fields[:10]:
            print(f"  - {field}: {count}")

    # Write Detailed Report
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("DATABASE ANALYSIS REPORT\n")
        f.write("========================\n\n")

        f.write("SUMMARY\n-------\n")
        f.write(f"Old File: {args.old_file}\n")
        f.write(f"New File: {args.new_file}\n\n")
        f.write(f"Total Old: {len(map_old)}\n")
        f.write(f"Total New: {len(map_new)}\n")
        f.write(f"Identical: {identical_count}\n")
        f.write(f"Removed:   {len(removed_keys)}\n")
        f.write(f"Added:     {len(added_keys)}\n")
        f.write(f"Changed:   {len(changed_entries)}\n")

        # Added Analysis (Years)
        if added_keys:
            added_entries = [map_new[uid] for uid in added_keys]
            analyze_added_years(added_entries, f)

        if removed_keys:
            f.write(
                f"\n\nREMOVED ENTRIES ({len(removed_keys)})\n-----------------------------------\n"
            )
            for uid in sorted(removed_keys):
                entry = map_old[uid]
                f.write(
                    f"- [{uid}] {entry.get('title')} ({entry.get('type')}, {entry.get('animeSeason', {}).get('year', 'N/A')})\n"
                )

        if added_keys:
            f.write(
                f"\n\nADDED ENTRIES DETAILED ({len(added_keys)})\n-----------------------------------\n"
            )
            for uid in sorted(added_keys):
                entry = map_new[uid]
                f.write(
                    f"- [{uid}] {entry.get('title')} ({entry.get('type')}, {entry.get('animeSeason', {}).get('year', 'N/A')})\n"
                )

        if changed_entries:
            f.write(
                f"\n\nCHANGED ENTRIES ({len(changed_entries)})\n-----------------------------------\n"
            )
            for uid, info in sorted(changed_entries.items()):
                f.write(f"ID: {uid}\nTitle: {info['title']}\n")
                for field, change in info["diff"].items():
                    f.write(f"  Field '{field}':\n")
                    f.write(
                        f"    - Old: {json.dumps(change['old'], ensure_ascii=False)}\n"
                    )
                    f.write(
                        f"    - New: {json.dumps(change['new'], ensure_ascii=False)}\n"
                    )
                f.write("\n")

    print(f"\nDetailed report written to: {args.report}")


if __name__ == "__main__":
    main()
