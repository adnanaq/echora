#!/usr/bin/env python3
"""
Trigger enrichment pipeline with anime entry from offline database.
Supports multiple input methods: index, title search, or custom file.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from enrichment.programmatic.enrichment_pipeline import (
    ProgrammaticEnrichmentPipeline,
)


def load_database(file_path: str) -> Dict[str, Any]:
    """Load anime database from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Database file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in database file: {e}")
        sys.exit(1)


def get_anime_by_index(database: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Get anime entry by index from database."""
    data = database.get("data", [])
    if 0 <= index < len(data):
        return data[index]
    else:
        print(f"Error: Index {index} out of range. Database has {len(data)} entries.")
        return None


def get_anime_by_title(database: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
    """Search for anime by title (case-insensitive, partial match)."""
    data = database.get("data", [])
    title_lower = title.lower()

    # Try exact match first
    for entry in data:
        if entry.get("title", "").lower() == title_lower:
            return entry

    # Try partial match
    matches = []
    for entry in data:
        entry_title = entry.get("title", "").lower()
        if title_lower in entry_title:
            matches.append(entry)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Error: Multiple matches found for '{title}':")
        for i, match in enumerate(matches[:10], 1):
            print(f"  {i}. {match.get('title')}")
        print("Please use --index with the specific entry number.")
        return None
    else:
        print(f"Error: No anime found with title containing '{title}'")
        return None


async def main():
    """
    Orchestrates command-line parsing, selection of an anime entry from an offline JSON database, and runs the enrichment pipeline for that entry.
    
    Parses CLI options (--file, --index, --title, --agent, --skip, --only), validates mutually exclusive and required arguments, loads the specified database file, selects an anime entry by index or title, and invokes ProgrammaticEnrichmentPipeline.enrich_anime with optional service filtering and agent directory. Prints progress, the selected entry summary, per-API success indicators, and total enrichment time. Exits the process with a non-zero status on argument validation failures or when the requested anime entry cannot be found.
    """
    parser = argparse.ArgumentParser(
        description="Run enrichment pipeline on anime from offline database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_enrichment.py --index 0                    # Process first anime in database
  python run_enrichment.py --title "One Piece"          # Search by title
  python run_enrichment.py --file custom.json --index 5 # Use custom database file

  # Service filtering
  python run_enrichment.py --title "Dandadan" --skip jikan anidb          # Skip slow services
  python run_enrichment.py --title "Dandadan" --only anime_planet         # Only fetch anime_planet

  # Agent directory control
  python run_enrichment.py --title "Dandadan" --agent "Dandadan_agent1"   # Use existing agent directory
  python run_enrichment.py --title "Dandadan" --agent "Dandadan_agent1" --only anime_planet  # Combine with filtering

Available services: jikan, anilist, kitsu, anidb, anime_planet, anisearch, animeschedule
        """
    )

    parser.add_argument(
        "--file",
        default="data/qdrant_storage/anime-offline-database.json",
        help="Path to anime database JSON file (default: data/qdrant_storage/anime-offline-database.json)"
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of anime entry in database (0-based)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Search for anime by title (case-insensitive)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Agent directory name (e.g., 'Dandadan_agent1'). If not provided, auto-generates with gap filling."
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="SERVICE",
        help="Skip specific services (e.g., --skip jikan anidb)"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SERVICE",
        help="Only fetch specific services (e.g., --only anime_planet anisearch)"
    )

    args = parser.parse_args()

    # Validate service filtering arguments
    if args.skip and args.only:
        print("Error: Cannot use both --skip and --only. Choose one.")
        sys.exit(1)

    # Validate arguments
    if args.index is None and args.title is None:
        print("Error: Must specify either --index or --title")
        parser.print_help()
        sys.exit(1)

    if args.index is not None and args.title is not None:
        print("Error: Cannot specify both --index and --title")
        sys.exit(1)

    # Load database
    print(f"Loading database: {args.file}")
    database = load_database(args.file)
    total_entries = len(database.get("data", []))
    print(f"Database loaded: {total_entries} anime entries")

    # Get anime entry
    if args.index is not None:
        anime_entry = get_anime_by_index(database, args.index)
    else:
        anime_entry = get_anime_by_title(database, args.title)

    if anime_entry is None:
        sys.exit(1)

    anime_title = anime_entry.get("title", "Unknown")
    print(f"\n{'='*60}")
    print(f"Processing: {anime_title}")
    print(f"Type: {anime_entry.get('type', 'Unknown')}")
    print(f"Episodes: {anime_entry.get('episodes', 'Unknown')}")
    print(f"Status: {anime_entry.get('status', 'Unknown')}")
    print(f"{'='*60}\n")

    async with ProgrammaticEnrichmentPipeline() as pipeline:
        # Run enrichment with optional service filtering and agent directory
        result = await pipeline.enrich_anime(
            anime_entry,
            agent_dir=args.agent,
            skip_services=args.skip,
            only_services=args.only
        )

        # Show results
        print(f"\n{'='*60}")
        print("API Results:")
        for api_name, data in result["api_data"].items():
            status = "✓" if data else "✗"
            print(f"  {api_name}: {status}")

        print(f"\nTime: {result['enrichment_metadata']['total_time']:.2f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
