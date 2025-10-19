#!/usr/bin/env python3
"""
Stage 4 Statistics Extraction Script

This script implements programmatic statistics extraction from multiple anime data sources
with proper normalization and standardization, replacing the LLM-based approach.

Key Features:
- Extract statistics from 6 data sources (MAL, AniList, AniDB, Anime-Planet, Kitsu, AnimeSchedule)
- Normalize all scores to 0-10 scale
- Handle missing fields gracefully
- Source-specific field mappings
- Standard StatisticsEntry format
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any


# Project root for resolving paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_source_data(temp_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all external source data files."""
    sources = {}

    source_files = {
        'jikan': f'{temp_dir}/jikan.json',
        'animeschedule': f'{temp_dir}/animeschedule.json',
        'kitsu': f'{temp_dir}/kitsu.json',
        'anime_planet': f'{temp_dir}/anime_planet.json',
        'anilist': f'{temp_dir}/anilist.json',
        'anidb': f'{temp_dir}/anidb.json',
    }

    for source_name, file_path in source_files.items():
        try:
            with open(file_path, 'r') as f:
                sources[source_name] = json.load(f)
                print(f"Loaded {source_name} data")
        except Exception as e:
            print(f"Warning: Could not load {source_name}: {e}")
            sources[source_name] = {}

    return sources


def safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dictionaries."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def normalize_score(score: Any, scale_factor: float = 1.0) -> Optional[float]:
    """
    Normalize score to 0-10 scale.

    Args:
        score: Raw score value
        scale_factor: Factor to apply (e.g., 0.1 for 0-100 scale, 2.0 for 0-5 scale)

    Returns:
        Normalized score as float or None
    """
    if score is None:
        return None

    try:
        score_float = float(score)
        if score_float == 0:
            return None
        normalized = score_float * scale_factor
        # Clamp to 0-10 range
        return max(0.0, min(10.0, normalized))
    except (ValueError, TypeError):
        return None


def extract_mal_statistics(jikan_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MAL statistics from Jikan data.

    Field mappings:
    - data.score → score (already 0-10 scale)
    - data.scored_by → scored_by
    - data.rank → rank
    - data.popularity → popularity_rank
    - data.members → members
    - data.favorites → favorites
    """
    data = safe_get(jikan_data, "data", default={})
    return {
        "score": normalize_score(safe_get(data, "score")),
        "scored_by": safe_get(data, "scored_by"),
        "rank": safe_get(data, "rank"),
        "popularity_rank": safe_get(data, "popularity"),
        "members": safe_get(data, "members"),
        "favorites": safe_get(data, "favorites")
    }


def extract_animeschedule_statistics(animeschedule_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AnimeSchedule statistics.

    Field mappings:
    - stats.averageScore → score (0-100 scale, divide by 10 for 0-10)
    - stats.ratingCount → scored_by
    - stats.trackedCount → members
    - stats.trackedRating → rank
    """
    stats = safe_get(animeschedule_data, "stats", default={})
    return {
        "score": normalize_score(safe_get(stats, "averageScore"), scale_factor=0.1),
        "scored_by": safe_get(stats, "ratingCount"),
        "rank": safe_get(stats, "trackedRating"),
        "popularity_rank": None,
        "members": safe_get(stats, "trackedCount"),
        "favorites": None
    }


def extract_kitsu_statistics(kitsu_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Kitsu statistics.

    Field mappings:
    - anime.attributes.averageRating → score (÷10 for 0-10 scale, e.g., "82.14" → 8.214)
    - anime.attributes.userCount → members
    - anime.attributes.favoritesCount → favorites
    - anime.attributes.popularityRank → popularity_rank
    - anime.attributes.ratingRank → rank
    """
    attributes = safe_get(kitsu_data, "anime", "attributes", default={})
    return {
        "score": normalize_score(safe_get(attributes, "averageRating"), scale_factor=0.1),
        "scored_by": None,
        "rank": safe_get(attributes, "ratingRank"),
        "popularity_rank": safe_get(attributes, "popularityRank"),
        "members": safe_get(attributes, "userCount"),
        "favorites": safe_get(attributes, "favoritesCount")
    }


def extract_animeplanet_statistics(animeplanet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Anime-Planet statistics.

    Field mappings:
    - aggregate_rating.ratingValue → score (×2 for 0-10 scale, e.g., 4.355 → 8.71)
    - aggregate_rating.ratingCount → scored_by
    - rank → rank
    """
    aggregate_rating = safe_get(animeplanet_data, "aggregate_rating", default={})
    return {
        "score": normalize_score(
            safe_get(aggregate_rating, "ratingValue"),
            scale_factor=2.0
        ),
        "scored_by": safe_get(aggregate_rating, "ratingCount"),
        "rank": safe_get(animeplanet_data, "rank")
    }


def extract_anilist_statistics(anilist_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AniList statistics.

    Field mappings:
    - averageScore → score (÷10 for 0-10 scale, e.g., 86 → 8.6)
    - favourites → favorites
    - popularity → popularity_rank
    - rankings → contextual_ranks (extract array of ranking objects)
    """
    # Extract contextual ranks
    contextual_ranks = None
    rankings = safe_get(anilist_data, "rankings")
    if rankings and isinstance(rankings, list) and len(rankings) > 0:
        contextual_ranks = []
        for rank_obj in rankings:
            if isinstance(rank_obj, dict):
                contextual_ranks.append({
                    "rank": rank_obj.get("rank"),
                    "type": rank_obj.get("type"),
                    "format": rank_obj.get("format"),
                    "year": rank_obj.get("year"),
                    "season": rank_obj.get("season"),
                    "all_time": rank_obj.get("allTime", False)
                })

    return {
        "score": normalize_score(safe_get(anilist_data, "averageScore"), scale_factor=0.1),
        "scored_by": None,
        "rank": None,
        "popularity_rank": safe_get(anilist_data, "popularity"),
        "members": None,
        "favorites": safe_get(anilist_data, "favourites"),
        "contextual_ranks": contextual_ranks
    }


def extract_anidb_statistics(anidb_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AniDB statistics.

    Field mappings:
    - ratings.permanent.value → score (already 0-10 scale)
    - ratings.permanent.count → scored_by
    """
    return {
        "score": normalize_score(safe_get(anidb_data, "ratings", "permanent", "value")),
        "scored_by": safe_get(anidb_data, "ratings", "permanent", "count"),
        "rank": None,
        "popularity_rank": None,
        "members": None,
        "favorites": None
    }


def has_any_statistics(stats: Dict[str, Any]) -> bool:
    """Check if statistics object has any non-null values (excluding contextual_ranks)."""
    return any(
        stats[key] is not None
        for key in ["score", "scored_by", "rank", "popularity_rank", "members", "favorites"]
    )


def extract_all_statistics(sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract statistics from all available sources.

    Returns:
        Dictionary with statistics organized by source
    """
    statistics = {}

    # Extract statistics from each source
    extractors = {
        "mal": ("jikan", extract_mal_statistics),
        "animeschedule": ("animeschedule", extract_animeschedule_statistics),
        "kitsu": ("kitsu", extract_kitsu_statistics),
        "animeplanet": ("anime_planet", extract_animeplanet_statistics),
        "anilist": ("anilist", extract_anilist_statistics),
        "anidb": ("anidb", extract_anidb_statistics),
    }

    for stat_key, (source_key, extractor_func) in extractors.items():
        source_data = sources.get(source_key, {})
        if source_data:
            stats = extractor_func(source_data)
            # Only include source if it has actual statistical data
            if has_any_statistics(stats):
                statistics[stat_key] = stats

    return statistics


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Extract statistics from multiple anime data sources"
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

    args = parser.parse_args()

    # Construct temp directory path
    temp_dir = f"{args.temp_dir}/{args.agent_id}"

    print(f"Stage 4: Statistics Extraction")
    print(f"Agent ID: {args.agent_id}")
    print(f"Temp directory: {temp_dir}")
    print("=" * 80)

    # Load source data
    print("\nLoading source data...")
    sources = load_source_data(temp_dir)

    # Extract statistics
    print("\nExtracting statistics from all sources...")
    statistics = extract_all_statistics(sources)

    # Print summary
    print("\nStatistics extraction summary:")
    for source in ["mal", "animeschedule", "kitsu", "animeplanet", "anilist", "anidb"]:
        if source in statistics:
            stats = statistics[source]
            fields = [k for k, v in stats.items() if v is not None and k != "contextual_ranks"]
            print(f"  {source}: {len(fields)} fields extracted")
        else:
            print(f"  {source}: No statistics available")

    # Save statistics to stage4 output file
    output_file = f"{temp_dir}/stage4_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"statistics": statistics}, f, indent=2, ensure_ascii=False)

    print(f"\nStatistics successfully saved to {output_file}")
    print("=" * 80)
    print("Stage 4 completed successfully!")


if __name__ == "__main__":
    main()
