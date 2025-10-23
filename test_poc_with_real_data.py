#!/usr/bin/env python3
"""Test Atomic Agents POC with Real Database Queries

This script tests the LLM's ability to parse various query patterns using
actual anime data from the enriched database.

Tests include:
- Single numerical filters (gte, lte, gt, lt)
- Multiple numerical filters combined
- Genre + score filters
- Type + status + score filters
- Complex multi-constraint queries
"""

import logging
import os
import sys

# Force CPU usage for embeddings to avoid CUDA OOM with Ollama
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.config.settings import get_settings
from src.poc.atomic_agents_poc import AnimeQueryAgent
from src.vector.client.qdrant_client import QdrantClient

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_separator(title: str) -> None:
    """Print a visual separator for test cases."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_results(result, original_query: str, max_results: int = 3) -> None:
    """Print search results in a readable format."""
    print(f'\nOriginal Query: "{original_query}"')
    print(f"Search Type: {result.search_type}")
    print(f"Results Count: {result.count}\n")

    if result.count > 0:
        print(f"Top {min(max_results, result.count)} Results:")
        for i, anime in enumerate(result.results[:max_results], 1):
            title = anime.get("title", "Unknown")
            score = anime.get("score", anime.get("_score", 0))

            # Extract statistics
            stats = anime.get("statistics", {})
            mal_score = stats.get("mal", {}).get("score", "N/A")
            anilist_score = stats.get("anilist", {}).get("score", "N/A")
            mal_members = stats.get("mal", {}).get("members", "N/A")

            # Extract metadata
            genres = anime.get("genres", [])
            anime_type = anime.get("type", "N/A")
            status = anime.get("status", "N/A")

            print(f"  {i}. {title}")
            print(f"     Similarity: {score:.4f}")
            print(
                f"     MAL Score: {mal_score}, AniList Score: {anilist_score}, Members: {mal_members}"
            )
            print(f"     Type: {anime_type}, Status: {status}")
            if genres:
                print(f"     Genres: {', '.join(genres[:5])}")
    else:
        print("No results found.")


def main() -> None:
    """Run comprehensive POC tests with real database queries."""

    # Initialize settings and Qdrant client
    print_separator("Initializing Atomic Agents POC")
    settings = get_settings()
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        settings=settings,
    )

    # Initialize agent with Ollama qwen3:30b
    agent = AnimeQueryAgent(
        qdrant_client=qdrant_client,
        model="qwen3:30b",
        ollama_base_url="http://localhost:11434/v1",
    )

    print("✓ Agent initialized with qwen3:30b model")
    print("✓ Qdrant client connected")
    print("✓ Ready to test queries\n")

    # Test Cases - Organized by complexity

    # ========================================================================
    # BASIC QUERIES - Single Filters
    # ========================================================================

    print_separator("TEST 1: Single Numerical Filter (gte)")
    query1 = "Find anime with MAL score greater than or equal to 7.0"
    result1 = agent.parse_and_search(query1)
    print_results(result1, query1)

    print_separator("TEST 2: Single Numerical Filter (lte)")
    query2 = "Show me anime with less than 1000 MAL members"
    result2 = agent.parse_and_search(query2)
    print_results(result2, query2)

    print_separator("TEST 3: Genre Filter Only")
    query3 = "Find comedy anime"
    result3 = agent.parse_and_search(query3)
    print_results(result3, query3)

    print_separator("TEST 4: Type Filter")
    query4 = "Show me movies"
    result4 = agent.parse_and_search(query4)
    print_results(result4, query4)

    # ========================================================================
    # INTERMEDIATE QUERIES - Multiple Filters
    # ========================================================================

    print_separator("TEST 5: Multiple Numerical Filters (MAL + AniList)")
    query5 = "Anime with MAL score above 6.5 and AniList score above 6.0"
    result5 = agent.parse_and_search(query5)
    print_results(result5, query5)

    print_separator("TEST 6: Genre + Score Filter")
    query6 = "Drama anime with MAL score greater than 7.0"
    result6 = agent.parse_and_search(query6)
    print_results(result6, query6)

    print_separator("TEST 7: Type + Score Filter")
    query7 = "Movies with MAL score above 6.8"
    result7 = agent.parse_and_search(query7)
    print_results(result7, query7)

    print_separator("TEST 8: Status + Type Filter")
    query8 = "Finished OVA series"
    result8 = agent.parse_and_search(query8)
    print_results(result8, query8)

    # ========================================================================
    # ADVANCED QUERIES - Complex Multi-Constraint
    # ========================================================================

    print_separator("TEST 9: Three Filters (Genre + Type + Score)")
    query9 = "Drama movies with MAL score above 7.2"
    result9 = agent.parse_and_search(query9)
    print_results(result9, query9)

    print_separator("TEST 10: Multiple Platforms + Genre")
    query10 = "Fantasy anime with MAL score above 6.0 and AniList score above 5.5"
    result10 = agent.parse_and_search(query10)
    print_results(result10, query10)

    print_separator("TEST 11: Popularity Filter (members)")
    query11 = "Anime with at least 900 MAL members"
    result11 = agent.parse_and_search(query11)
    print_results(result11, query11)

    print_separator("TEST 12: Range Filter (between scores)")
    query12 = "Anime with MAL score between 5.5 and 6.5"
    result12 = agent.parse_and_search(query12)
    print_results(result12, query12)

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    print_separator("TEST 13: Multiple Genres + Score")
    query13 = "Comedy or Drama anime with AniList score above 6.5"
    result13 = agent.parse_and_search(query13)
    print_results(result13, query13)

    print_separator("TEST 14: Complex Natural Language")
    query14 = "I want to find highly rated finished movies in the drama or psychological genre"
    result14 = agent.parse_and_search(query14)
    print_results(result14, query14)

    print_separator("TEST 15: Specific Platform Scores")
    query15 = "Anime with AnimePlanet score above 6.3"
    result15 = agent.parse_and_search(query15)
    print_results(result15, query15)

    # ========================================================================
    # Summary
    # ========================================================================

    print_separator("TEST SUMMARY")
    print("✓ Tested 15 different query patterns")
    print("✓ Covered single filters, multiple filters, and complex constraints")
    print("✓ Tested numerical ranges (gte, lte, gt, lt)")
    print("✓ Tested genre, type, status, and multi-platform score filters")
    print("\nCheck the logs above to verify:")
    print("  - LLM correctly parsed each query")
    print("  - Filters were properly formatted")
    print("  - Qdrant Filter objects show correct Range values")
    print("  - Results match the filter criteria")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
