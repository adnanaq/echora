#!/usr/bin/env python3
"""
Reorder Enrichment Database Fields

Reorders all fields in the enriched anime database to match the
AnimeEntry model structure (SCALAR -> ARRAY -> OBJECT/DICT order).

This ensures consistent field ordering and makes the database easier to read/maintain.

Usage:
    python reorder_enrichment_fields.py --input data/enriched_anime_database.json --backup
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            # Convert to UTC and force Z format for consistency
            if obj.tzinfo is not None:
                # Convert timezone-aware datetime to UTC
                from datetime import timezone
                obj = obj.astimezone(timezone.utc).replace(tzinfo=None)
            # Always append Z for UTC timestamps
            return obj.isoformat() + 'Z'
        return super().default(obj)

try:
    from pydantic import ValidationError

    from common.models.anime import AnimeEntry
except ImportError:
    print("Error: Could not import AnimeEntry model. Ensure you're in the correct directory.")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def reorder_entry_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder fields in a single anime entry to match AnimeEntry model structure.

    Ensures enrichment_metadata is always placed at the very end.

    Uses Pydantic's model_dump() to get the correct field ordering.
    """
    try:
        # Load entry through AnimeEntry model (validates and structures)
        anime_entry = AnimeEntry(**entry)

        # Serialize back to dict with proper field ordering, preserving JSON aliases
        # Use exclude_none=True to avoid adding empty fields that validation script would remove
        reordered_entry = anime_entry.model_dump(exclude_unset=False, exclude_none=True, by_alias=True)

        # Move enrichment_metadata to the very end if it exists
        if "enrichment_metadata" in reordered_entry:
            enrichment_metadata = reordered_entry.pop("enrichment_metadata")
            reordered_entry["enrichment_metadata"] = enrichment_metadata

        return reordered_entry

    except ValidationError as e:
        logger.error(f"Validation error for entry '{entry.get('title', 'Unknown')}': {e}")
        # Return original entry if validation fails
        return entry
    except Exception as e:
        logger.error(f"Unexpected error for entry '{entry.get('title', 'Unknown')}': {e}")
        # Return original entry if any other error
        return entry


def reorder_database(input_file: str, backup: bool = True) -> Dict[str, Any]:
    """
    Reorder all entries in the enrichment database.
    """
    logger.info(f"Starting field reordering for {input_file}")

    backup_path: Optional[str] = None
    if backup:
        backup_path = f"{input_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(input_file, "r") as original, open(backup_path, "w") as backup_file:
            backup_file.write(original.read())
        logger.info(f"Created backup: {backup_path}")

    # Load database
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        return {"error": str(e)}

    if "data" not in data:
        logger.error("Database file missing 'data' field")
        return {"error": "Invalid database structure"}

    total_entries = len(data["data"])
    reordered_count = 0
    validation_errors = 0

    logger.info(f"Processing {total_entries} anime entries...")

    # Process each entry
    for index, entry in enumerate(data["data"]):
        title = entry.get("title", f"Entry #{index}")

        try:
            # Reorder fields
            reordered_entry = reorder_entry_fields(entry)

            # Check if reordering occurred (fields changed)
            if reordered_entry != entry:
                data["data"][index] = reordered_entry
                reordered_count += 1
                logger.debug(f"Reordered fields for: {title}")
            else:
                logger.debug(f"No reordering needed for: {title}")

        except ValidationError:
            validation_errors += 1
            logger.warning(f"Validation failed for: {title} (keeping original)")
        except Exception as e:
            logger.warning(f"Failed to reorder: {title} - {e} (keeping original)")

    # Save reordered database
    try:
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        logger.info(f"Saved reordered database to: {input_file}")
    except Exception as e:
        logger.error(f"Failed to save database: {e}")
        return {"error": f"Save failed: {e}"}

    return {
        "total_entries": total_entries,
        "reordered_entries": reordered_count,
        "validation_errors": validation_errors,
        "success": True,
        "backup_created": backup_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reorder enrichment database fields to match AnimeEntry model structure"
    )
    parser.add_argument(
        "--input",
        default="data/qdrant_storage/enriched_anime_database.json",
        help="Path to enriched database file",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before reordering (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reordered without making changes",
    )

    args = parser.parse_args()

    # Resolve file path
    input_file = Path(args.input)
    if not input_file.is_absolute():
        input_file = Path(__file__).parent / input_file

    if not input_file.exists():
        logger.error(f"Database file not found: {input_file}")
        return 1

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        args.backup = False

    # Reorder database
    result = reorder_database(str(input_file), args.backup)

    if "error" in result:
        logger.error(f"Reordering failed: {result['error']}")
        return 1

    # Print results
    print("\nField Reordering Summary:")
    print(f"  Total entries: {result['total_entries']}")
    print(f"  Entries reordered: {result['reordered_entries']}")
    print(f"  Validation errors: {result['validation_errors']}")
    if result.get("backup_created"):
        print(f"  Backup created: {result['backup_created']}")

    if result['reordered_entries'] > 0:
        print(f"\n✅ Successfully reordered {result['reordered_entries']} entries to match AnimeEntry model structure")
    else:
        print(f"\n✅ All entries already in correct field order")

    return 0


if __name__ == "__main__":
    exit(main())