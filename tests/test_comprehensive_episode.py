#!/usr/bin/env python3
"""
Comprehensive Episode Vector Validation Test

Tests episode search functionality with random field combinations and comprehensive validation.
Follows the same pattern as test_comprehensive_title.py and test_character_vector.py.
"""

import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this file as integration tests (require Qdrant running)
pytestmark = pytest.mark.integration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.vector.client.qdrant_client import QdrantClient
from src.vector.processors.text_processor import TextProcessor


def load_episode_data() -> dict[str, list[dict]]:
    """Load episode data from enrichment file."""
    with open(
        "./data/qdrant_storage/enriched_anime_database.json", encoding="utf-8"
    ) as f:
        data = json.load(f)

    anime_with_episodes = {}
    for anime in data["data"]:
        anime_title = anime["title"]
        episode_details = anime.get("episode_details", [])

        if episode_details and len(episode_details) > 0:
            anime_with_episodes[anime_title] = episode_details

    return anime_with_episodes


def get_episode_fields(episode: dict[str, Any]) -> list[str]:
    """Get available fields from an episode for testing."""
    available_fields = []

    # Check each field and include only if it has meaningful content
    if episode.get("title"):
        available_fields.append("title")
    if episode.get("title_japanese"):
        available_fields.append("title_japanese")
    if episode.get("title_romaji"):
        available_fields.append("title_romaji")
    if episode.get("synopsis"):
        available_fields.append("synopsis")
    if episode.get("aired"):
        available_fields.append("aired")
    if episode.get("score"):
        available_fields.append("score")
    if episode.get("duration"):
        available_fields.append("duration")
    if episode.get("episode_number"):
        available_fields.append("episode_number")
    if episode.get("season_number"):
        available_fields.append("season_number")
    if episode.get("filler") is not None:
        available_fields.append("filler")
    if episode.get("recap") is not None:
        available_fields.append("recap")

    return available_fields


def generate_field_combinations(fields: list[str]) -> list[list[str]]:
    """Generate all possible field combinations for comprehensive testing."""
    combinations_list = []

    # Single fields
    for field in fields:
        combinations_list.append([field])

    # Pairs
    for pair in combinations(fields, 2):
        combinations_list.append(list(pair))

    # Triples
    for triple in combinations(fields, 3):
        combinations_list.append(list(triple))

    # Larger combinations (up to 5 fields to keep it manageable)
    for size in [4, 5]:
        if len(fields) >= size:
            for combo in combinations(fields, min(size, len(fields))):
                combinations_list.append(list(combo))

    # Full combination if reasonable
    if len(fields) <= 8:
        combinations_list.append(fields)

    return combinations_list


def create_episode_query(episode: dict[str, Any], field_combination: list[str]) -> str:
    """Create a search query from episode data using specified fields."""
    query_parts = []

    for field in field_combination:
        value = episode.get(field)
        if value is not None:
            if field == "title":
                query_parts.append(str(value))
            elif field == "title_japanese":
                query_parts.append(str(value))
            elif field == "title_romaji":
                query_parts.append(str(value))
            elif field == "synopsis":
                # Truncate synopsis to keep query manageable
                synopsis = str(value)[:150]
                query_parts.append(synopsis)
            elif field == "aired":
                # Convert date to more searchable format
                aired_str = str(value)
                if "T" in aired_str:
                    date_part = aired_str.split("T")[0]
                    query_parts.append(f"aired {date_part}")
                else:
                    query_parts.append(f"aired {aired_str}")
            elif field == "score":
                query_parts.append(f"rated {value}")
            elif field == "duration":
                # Convert duration to minutes for more natural search
                duration_minutes = int(value) // 60 if isinstance(value, int) else value
                query_parts.append(f"{duration_minutes} minutes")
            elif field == "episode_number":
                query_parts.append(f"episode {value}")
            elif field == "season_number":
                query_parts.append(f"season {value}")
            elif field == "filler":
                if value:
                    query_parts.append("filler episode")
            elif field == "recap":
                if value:
                    query_parts.append("recap episode")

    return " ".join(query_parts).strip()


def print_episode_info(episode: dict[str, Any]) -> str:
    """Format episode information for display."""
    info_parts = []

    if episode.get("episode_number"):
        info_parts.append(f"Ep.{episode['episode_number']}")
    if episode.get("title"):
        title = episode["title"]
        if len(title) > 40:
            title = title[:37] + "..."
        info_parts.append(f'"{title}"')
    if episode.get("aired"):
        aired = str(episode["aired"]).split("T")[0]
        info_parts.append(f"({aired})")

    return " ".join(info_parts)


async def test_comprehensive_episode_vector():
    """
    Comprehensive episode vector validation with random selection and field combinations.
    """
    print(
        "╔══════════════════════════════════════════════════════════════════════════════╗"
    )
    print(
        "║                                                                              ║"
    )
    print(
        "║  🎬 Comprehensive Episode Vector Validation                                  ║"
    )
    print(
        "║  Testing episode search with random field combinations                       ║"
    )
    print(
        "║                                                                              ║"
    )
    print(
        "╚══════════════════════════════════════════════════════════════════════════════╝"
    )

    settings = get_settings()
    qdrant_client = QdrantClient(settings=settings)
    text_processor = TextProcessor(settings=settings)

    # Load episode data
    anime_with_episodes = load_episode_data()

    if not anime_with_episodes:
        print("❌ No anime with episode details found for testing")
        return

    # True randomization with timestamp seed
    seed = int(time.time() * 1000) % 2**32
    random.seed(seed)

    total_anime = len(anime_with_episodes)
    test_count = min(5, total_anime)

    print(
        "╭────────────────────── Episode Vector Test Configuration ──────────────────────╮"
    )
    print(
        "│ ╭─────────────────────────────┬────────────╮                                 │"
    )
    print(
        f"│ │ 📊 Total anime with episodes: │ {total_anime:<10} │                                 │"
    )
    print(
        f"│ │ 🎯 Testing anime count:       │ {test_count:<10} │                                 │"
    )
    print(
        f"│ │ 🎲 Random seed:               │ {seed:<10} │                                 │"
    )
    print(
        "│ ╰─────────────────────────────┴────────────╯                                 │"
    )
    print(
        "╰──────────────────────────────────────────────────────────────────────────────╯"
    )

    # Randomly select anime for testing
    test_anime = random.sample(list(anime_with_episodes.keys()), test_count)

    total_tests = 0
    passed_tests = 0
    test_details = []

    for i, anime_title in enumerate(test_anime, 1):
        print(f"\nTest #{i} {'═' * (70 - len(str(i)))}")
        print()

        episodes = anime_with_episodes[anime_title]

        # Randomly select an episode
        selected_episode = random.choice(episodes)

        print("📺 Target Anime")
        print(f"   {anime_title}")

        episode_info = print_episode_info(selected_episode)
        print("📝 Selected Episode")
        print(f"   {episode_info}")

        # Get available fields for this episode
        available_fields = get_episode_fields(selected_episode)

        if not available_fields:
            print("   ⚠️  No testable fields available, skipping...")
            continue

        # Generate field combinations
        field_combinations = generate_field_combinations(available_fields)

        # Randomly select a field combination
        selected_combination = random.choice(field_combinations)

        # Display field information
        field_info = []
        for field in available_fields:
            value = selected_episode.get(field)
            if field == "title":
                field_info.append(f'{field}: "{value}" ({len(str(value))} characters)')
            elif field == "synopsis":
                synopsis = str(value)
                char_count = len(synopsis)
                field_info.append(
                    f"{field}: {synopsis[:50]}{'...' if len(synopsis) > 50 else ''} ({char_count} characters)"
                )
            elif field in ["title_japanese", "title_romaji"]:
                field_info.append(f"{field}: {value} ({len(str(value))} characters)")
            else:
                field_info.append(f"{field}: {value}")

        print(
            "╭──────────────────────────── 📋 Available Fields ─────────────────────────────╮"
        )
        for info in field_info:
            print(f"│ {info:<76} │")
        print(
            "╰──────────────────────────────────────────────────────────────────────────────╯"
        )

        print(
            "╭─────────────────────────── 📋 Field Combinations ────────────────────────────╮"
        )
        combination_str = " + ".join(selected_combination)
        print(f"│ {combination_str:<76} │")
        print(
            "╰──────────────────────────────────────────────────────────────────────────────╯"
        )

        # Create query
        query = create_episode_query(selected_episode, selected_combination)

        print(
            "╭────────────────────────── 📝 Generated Text Query ───────────────────────────╮"
        )
        query_lines = [query[i : i + 76] for i in range(0, len(query), 76)]
        for line in query_lines:
            print(f"│ {line:<76} │")
        print(
            "╰──────────────────────────────────────────────────────────────────────────────╯"
        )

        if not query.strip():
            print("   ⚠️  Empty query generated, skipping...")
            continue

        try:
            # Generate embedding for the query
            embedding = text_processor.encode_text(query)
            if not embedding:
                print("   ❌ Failed to generate embedding")
                continue

            # Search using episode vector
            results = await qdrant_client.search_single_vector(
                vector_name="episode_vector", vector_data=embedding, limit=5
            )

            print("\n📊 Search Results")
            print()

            if results:
                # Format results table
                print(
                    "    #   Title                                               Score    Status   "
                )
                print(
                    " ──────────────────────────────────────────────────────────────────────────── "
                )

                found_target = False
                for j, result in enumerate(results):
                    result_title = result.get("title", "Unknown")
                    score = result.get("score", 0.0)

                    # Truncate title if too long
                    display_title = result_title
                    if len(display_title) > 47:
                        display_title = display_title[:44] + "..."

                    status = ""
                    if result_title == anime_title:
                        status = "✅ PASS"
                        if j == 0:  # Top result
                            found_target = True

                    print(
                        f"    {j + 1:<3} {display_title:<47} {score:<8.4f} {status:<8}"
                    )

                print()

                # Analysis
                top_result = results[0]
                top_title = top_result.get("title", "Unknown")
                top_score = top_result.get("score", 0.0)

                if found_target:
                    print(
                        "🎯 Analysis: EXACT MATCH - Perfect episode semantic similarity!"
                    )
                    passed_tests += 1
                    test_details.append(
                        {
                            "anime": anime_title,
                            "episode": episode_info,
                            "combination": combination_str,
                            "query": query[:50] + "..." if len(query) > 50 else query,
                            "score": top_score,
                            "status": "PASS",
                        }
                    )
                else:
                    print(
                        f"🎯 Analysis: NO MATCH - Expected '{anime_title}', got '{top_title}'"
                    )
                    test_details.append(
                        {
                            "anime": anime_title,
                            "episode": episode_info,
                            "combination": combination_str,
                            "query": query[:50] + "..." if len(query) > 50 else query,
                            "score": top_score,
                            "status": "FAIL",
                        }
                    )

                total_tests += 1

            else:
                print("   ❌ No search results returned")
                total_tests += 1
                test_details.append(
                    {
                        "anime": anime_title,
                        "episode": episode_info,
                        "combination": combination_str,
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "score": 0.0,
                        "status": "NO RESULTS",
                    }
                )

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            total_tests += 1
            test_details.append(
                {
                    "anime": anime_title,
                    "episode": episode_info,
                    "combination": combination_str,
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "score": 0.0,
                    "status": "ERROR",
                }
            )

        print("═" * 79)

    # Summary results
    print()
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100

        print("                    Field Combination Analysis                    ")
        print("                                                                  ")
        print("  Combination              T     P    Success    Score    Type    ")
        print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ")

        for detail in test_details[:4]:  # Show first 4 for table
            combo_short = (
                detail["combination"][:20] + "..."
                if len(detail["combination"]) > 23
                else detail["combination"]
            )
            status_char = "✅" if detail["status"] == "PASS" else "❌"
            success_pct = "100%" if detail["status"] == "PASS" else "0%"

            print(
                f"  {combo_short:<23}     1     {1 if detail['status'] == 'PASS' else 0}       {success_pct:<7}    {detail['score']:.3f}            "
            )

        print()

        print(
            "╭───────────────────────── Episode Vector Test Results ────────────────────────╮"
        )
        print(
            "│ ╭──────────────────┬────────────╮                                            │"
        )
        print(
            f"│ │ 🎲 Random seed:  │ {seed:<10} │                                            │"
        )
        print(
            f"│ │ ✅ Tests passed: │ {passed_tests}/{total_tests:<9} │                                            │"
        )
        print(
            f"│ │ 📊 Success rate: │ {success_rate:.1f}% {'🎉' if success_rate == 100 else '⚠️' if success_rate >= 60 else '❌':<8} │                                            │"
        )

        assessment = (
            "EXCELLENT"
            if success_rate >= 90
            else "GOOD" if success_rate >= 70 else "NEEDS WORK"
        )
        print(
            f"│ │ 🔬 Assessment:   │ {assessment:<10} │                                            │"
        )
        print(
            "│ ╰──────────────────┴────────────╯                                            │"
        )
        print(
            "╰──────────────────────────────────────────────────────────────────────────────╯"
        )

        print(
            "╭────────────────────────────── 🎯 Key Insights ───────────────────────────────╮"
        )
        print(
            "│ • Random episode selection ensures comprehensive testing coverage             │"
        )
        print(
            "│ • Field combination testing validates semantic understanding of episodes     │"
        )
        print(
            "│ • Production search method used for real-world validation                   │"
        )
        print(
            "│ • Episode-specific fields tested: titles, synopsis, metadata, flags        │"
        )
        print(
            "╰──────────────────────────────────────────────────────────────────────────────╯"
        )
    else:
        print("❌ No episode vector tests completed")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_comprehensive_episode_vector())
