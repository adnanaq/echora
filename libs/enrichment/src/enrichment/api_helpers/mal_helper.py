#!/usr/bin/env python3
"""
MAL API helper entrypoint.

SRP:
- Re-export `MalEnrichmentHelper` for programmatic use.
- Provide a small CLI for fetching MAL data to JSON files.

CLI:
  python -m enrichment.api_helpers.mal_helper anime <mal_id> <output_file>
  python -m enrichment.api_helpers.mal_helper episodes <mal_id> <episode_count> <output_file>
  python -m enrichment.api_helpers.mal_helper characters <mal_id> <output_file>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from enrichment.api_helpers.mal_enrichment_helper import MalEnrichmentHelper


def _write_json_sync(path: str, data: Any) -> None:
    """Write a JSON document to disk synchronously.

    This helper is intended for CLI use. Service paths should avoid blocking disk I/O.

    Args:
        path: Output path to write.
        data: JSON-serializable data to write.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main() -> int:
    """CLI entrypoint for fetching MAL data.

    Returns:
        Process exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Fetch data from the MAL API")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anime = sub.add_parser("anime", help="Fetch anime /full payload")
    p_anime.add_argument("anime_id", help="MyAnimeList anime ID")
    p_anime.add_argument("output_file", help="Output JSON file")

    p_eps = sub.add_parser("episodes", help="Fetch detailed episode payloads")
    p_eps.add_argument("anime_id", help="MyAnimeList anime ID")
    p_eps.add_argument(
        "episode_count", type=int, help="Number of episodes to fetch (1..N)"
    )
    p_eps.add_argument("output_file", help="Output JSON file")

    p_chars = sub.add_parser("characters", help="Fetch detailed character payloads")
    p_chars.add_argument("anime_id", help="MyAnimeList anime ID")
    p_chars.add_argument("output_file", help="Output JSON file")

    args = parser.parse_args()

    async with MalEnrichmentHelper(args.anime_id) as helper:
        if args.cmd == "anime":
            data = await helper.fetch_anime()
            _write_json_sync(args.output_file, data or {})
            return 0

        if args.cmd == "episodes":
            episodes = await helper.fetch_episodes(args.episode_count)
            _write_json_sync(args.output_file, episodes)
            return 0

        if args.cmd == "characters":
            basic = await helper.fetch_characters_basic()
            detailed = await helper.fetch_characters_detailed(basic)
            _write_json_sync(args.output_file, detailed)
            return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
