#!/usr/bin/env python3
"""
Final Assembly Script - Merge All Enrichment Stages

This script implements Step 5: Programmatic Assembly from enrichment_instructions.md.
It merges all 6 stage outputs into a single enriched anime object following the
AnimeEntry schema with proper field ordering and data quality cleanup.

Key Features:
- Synchronization check for all 6 stage output files
- Merge stage outputs following AnimeEntry schema order
- Data quality cleanup (remove null values and empty strings)
- Unicode character handling with ensure_ascii=False
- enrichment_metadata as final field
- Leverages existing reorder_entry_fields() from reorder_enrichment_fields.py

Usage:
    python scripts/assemble_enrichment.py <agent_id> [--temp-dir <base>]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import reorder function from existing script
from reorder_enrichment_fields import reorder_entry_fields

# Project root for resolving paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_stage_files(temp_dir: str) -> bool:
    """
    Verify all 6 stage output files exist.

    Args:
        temp_dir: Agent's temp directory path

    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        "stage1_metadata.json",
        "stage2_episodes.json",
        "stage3_relationships.json",
        "stage4_statistics.json",
        "stage5_characters.json",
        "stage6_staff.json"
    ]

    missing_files = []
    for filename in required_files:
        file_path = Path(temp_dir) / filename
        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        print(f"Error: Missing stage output files:")
        for filename in missing_files:
            print(f"  - {filename}")
        return False

    return True


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON file with error handling.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data as dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def remove_null_and_empty_strings(data: Any, preserve_required: bool = False,
                                   required_fields: Optional[List[str]] = None) -> Any:
    """
    Recursively remove null values and empty strings from data structures.
    Preserves empty arrays [] and empty objects {}.

    Args:
        data: Data structure to clean
        preserve_required: Whether to preserve required fields even if empty
        required_fields: List of required field names to preserve

    Returns:
        Cleaned data structure
    """
    if required_fields is None:
        required_fields = ["title", "status", "type", "episodes", "sources"]

    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Preserve required fields
            if preserve_required and key in required_fields:
                cleaned[key] = remove_null_and_empty_strings(value, preserve_required, required_fields)
                continue

            # Skip null values
            if value is None:
                continue

            # Skip empty strings
            if isinstance(value, str) and value == "":
                continue

            # Preserve empty collections
            if isinstance(value, (list, dict)):
                cleaned[key] = remove_null_and_empty_strings(value, preserve_required, required_fields)
                continue

            # Keep all other values
            cleaned[key] = remove_null_and_empty_strings(value, preserve_required, required_fields)

        return cleaned

    elif isinstance(data, list):
        return [remove_null_and_empty_strings(item, preserve_required, required_fields) for item in data]

    else:
        return data


def merge_stage_outputs(temp_dir: str) -> Dict[str, Any]:
    """
    Merge all 6 stage outputs into a single enriched anime object.

    Args:
        temp_dir: Agent's temp directory path

    Returns:
        Merged anime data dictionary
    """
    temp_path = Path(temp_dir)

    # Load current anime data (original offline_anime_data)
    current_anime = load_json_file(temp_path / "current_anime.json")

    # Start with original data
    enriched_anime = current_anime.copy()

    # Load all stage outputs
    stage1 = load_json_file(temp_path / "stage1_metadata.json")
    stage2 = load_json_file(temp_path / "stage2_episodes.json")
    stage3 = load_json_file(temp_path / "stage3_relationships.json")
    stage4 = load_json_file(temp_path / "stage4_statistics.json")
    stage5 = load_json_file(temp_path / "stage5_characters.json")
    stage6 = load_json_file(temp_path / "stage6_staff.json")

    # Merge Stage 1: Metadata
    if stage1:
        enriched_anime.update(stage1)

    # Merge Stage 2: Episodes
    if "episode_details" in stage2:
        enriched_anime["episode_details"] = stage2["episode_details"]

    # Merge Stage 3: Relationships
    if "relations" in stage3:
        enriched_anime["relations"] = stage3["relations"]
    if "related_anime" in stage3:
        enriched_anime["related_anime"] = stage3["related_anime"]

    # Merge Stage 4: Statistics
    if "statistics" in stage4:
        enriched_anime["statistics"] = stage4["statistics"]

    # Merge Stage 5: Characters
    if "characters" in stage5:
        enriched_anime["characters"] = stage5["characters"]

    # Merge Stage 6: Staff
    if "staff_data" in stage6:
        enriched_anime["staff_data"] = stage6["staff_data"]

    return enriched_anime


def add_enrichment_metadata(anime_data: Dict[str, Any], success: bool = True,
                            error_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Add enrichment_metadata as the VERY LAST field.

    Args:
        anime_data: Anime data dictionary
        success: Whether enrichment was successful
        error_message: Error message if enrichment failed

    Returns:
        Anime data with enrichment_metadata
    """
    enrichment_metadata = {
        "source": "multi-source",
        "enriched_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "success": success,
        "error_message": error_message
    }

    anime_data["enrichment_metadata"] = enrichment_metadata
    return anime_data


def assemble_enrichment(temp_dir: str, output_file: Optional[str] = None) -> bool:
    """
    Main assembly function that orchestrates the entire process.

    Args:
        temp_dir: Agent's temp directory path
        output_file: Optional output file path (default: temp_dir/enriched_anime.json)

    Returns:
        True if assembly successful, False otherwise
    """
    print(f"Final Assembly: Merging all enrichment stages")
    print(f"Temp directory: {temp_dir}")
    print("=" * 80)

    # Step 1: Synchronization check
    print("\nStep 1: Checking for all stage output files...")
    if not check_stage_files(temp_dir):
        return False
    print("✓ All stage files present")

    # Step 2: Merge all stages
    print("\nStep 2: Merging stage outputs...")
    try:
        enriched_anime = merge_stage_outputs(temp_dir)
        print(f"✓ Merged {len(enriched_anime)} top-level fields")
    except Exception as e:
        print(f"Error during merge: {e}")
        return False

    # Step 3: Data quality cleanup
    print("\nStep 3: Data quality cleanup...")
    enriched_anime = remove_null_and_empty_strings(enriched_anime, preserve_required=True)
    print("✓ Removed null values and empty strings")

    # Step 4: Add enrichment_metadata
    print("\nStep 4: Adding enrichment metadata...")
    enriched_anime = add_enrichment_metadata(enriched_anime, success=True)
    print("✓ Added enrichment_metadata")

    # Step 5: Order fields by schema using existing reorder function
    print("\nStep 5: Ordering fields by AnimeEntry schema...")
    try:
        enriched_anime = reorder_entry_fields(enriched_anime)
        print("✓ Fields ordered (SCALAR -> ARRAY -> OBJECT, enrichment_metadata last)")
    except Exception as e:
        print(f"Warning: Field reordering failed: {e}")
        print("Continuing with current field order...")

    # Step 6: Save output
    if output_file is None:
        output_file = f"{temp_dir}/enriched_anime.json"

    print(f"\nStep 6: Saving enriched anime to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_anime, f, indent=2, ensure_ascii=False)
        print("✓ Enriched anime saved successfully")
    except Exception as e:
        print(f"Error saving output: {e}")
        return False

    print("\n" + "=" * 80)
    print("Final Assembly completed successfully!")
    return True


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Final Assembly: Merge all enrichment stages into a single anime object"
    )
    parser.add_argument(
        "agent_id",
        help="Agent ID (directory name in temp/)"
    )
    parser.add_argument(
        "--temp-dir",
        default="temp",
        help="Base temp directory (default: temp)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: temp/<agent_id>/enriched_anime.json)"
    )

    args = parser.parse_args()

    # Construct temp directory path
    temp_dir = f"{args.temp_dir}/{args.agent_id}"

    print(f"Final Assembly - Enrichment Pipeline")
    print(f"Agent ID: {args.agent_id}")
    print(f"Temp directory: {temp_dir}")
    print("=" * 80)

    # Run assembly
    success = assemble_enrichment(temp_dir, args.output)

    if not success:
        print("\nFinal Assembly failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
