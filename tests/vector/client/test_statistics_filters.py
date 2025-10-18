"""Comprehensive test for all statistics filters using QdrantClient infrastructure."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path("/home/dani/code/anime-vector-service")
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient


async def test_filter(
    client: QdrantClient,
    filter_dict: Dict[str, Any],
    description: str,
    expected_fields: List[str]
) -> Dict[str, Any]:
    """Test a single filter using QdrantClient infrastructure.

    Args:
        client: QdrantClient instance
        filter_dict: Filter dictionary to pass to _build_filter()
        description: Human-readable description of the test
        expected_fields: Payload fields to include in results

    Returns:
        Test result dictionary with status and sample data
    """
    try:
        # Use QdrantClient's _build_filter() method
        qdrant_filter = client._build_filter(filter_dict)

        # Use the underlying Qdrant SDK client's scroll method
        results, _ = client.client.scroll(
            collection_name=client.collection_name,
            scroll_filter=qdrant_filter,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        points = results if isinstance(results, list) else []

        return {
            "description": description,
            "filter_dict": filter_dict,
            "status": "PASS",
            "result_count": len(points),
            "sample_results": [
                {
                    "title": p.payload.get("title"),
                    "values": {field: _extract_nested_value(p.payload, field) for field in expected_fields if field != "title"}
                }
                for p in points[:3]
            ]
        }
    except Exception as e:
        return {
            "description": description,
            "filter_dict": filter_dict,
            "status": f"FAIL: {str(e)}",
            "result_count": 0,
            "sample_results": []
        }


def _extract_nested_value(obj: Dict[str, Any], key_path: str) -> Any:
    """Extract value from nested dict using dot notation."""
    keys = key_path.split(".")
    value = obj
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


async def run_comprehensive_tests():
    """Run comprehensive tests on all statistics filters using QdrantClient."""

    settings = get_settings()

    # Initialize QdrantClient (same way as main.py)
    client = QdrantClient(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        settings=settings,
    )

    print("COMPREHENSIVE STATISTICS FILTER TESTING (Using QdrantClient)")
    print("=" * 80)
    print()

    # Define all test cases with proper filter dict format
    test_cases = [
        # MAL Statistics (6 fields)
        ({"statistics.mal.score": {"gte": 7.0}}, "MAL score >= 7.0", ["title", "statistics.mal.score"]),
        ({"statistics.mal.score": {"gte": 6.0, "lte": 8.0}}, "MAL score 6.0-8.0", ["title", "statistics.mal.score"]),
        ({"statistics.mal.scored_by": {"gte": 10000}}, "MAL scored_by >= 10K", ["title", "statistics.mal.scored_by"]),
        ({"statistics.mal.members": {"gte": 50000}}, "MAL members >= 50K", ["title", "statistics.mal.members"]),
        ({"statistics.mal.favorites": {"gte": 100}}, "MAL favorites >= 100", ["title", "statistics.mal.favorites"]),
        ({"statistics.mal.rank": {"lte": 5000}}, "MAL rank <= 5000", ["title", "statistics.mal.rank"]),
        ({"statistics.mal.popularity_rank": {"lte": 10000}}, "MAL popularity_rank <= 10K", ["title", "statistics.mal.popularity_rank"]),

        # AniList Statistics (3 fields)
        ({"statistics.anilist.score": {"gte": 7.0}}, "AniList score >= 7.0", ["title", "statistics.anilist.score"]),
        ({"statistics.anilist.favorites": {"gte": 10}}, "AniList favorites >= 10", ["title", "statistics.anilist.favorites"]),
        ({"statistics.anilist.popularity_rank": {"lte": 5000}}, "AniList popularity_rank <= 5000", ["title", "statistics.anilist.popularity_rank"]),

        # AniDB Statistics (2 fields)
        ({"statistics.anidb.score": {"gte": 7.0}}, "AniDB score >= 7.0", ["title", "statistics.anidb.score"]),
        ({"statistics.anidb.scored_by": {"gte": 1000}}, "AniDB scored_by >= 1000", ["title", "statistics.anidb.scored_by"]),

        # Anime-Planet Statistics (3 fields)
        ({"statistics.animeplanet.score": {"gte": 7.0}}, "Anime-Planet score >= 7.0", ["title", "statistics.animeplanet.score"]),
        ({"statistics.animeplanet.scored_by": {"gte": 1000}}, "Anime-Planet scored_by >= 1000", ["title", "statistics.animeplanet.scored_by"]),
        ({"statistics.animeplanet.rank": {"lte": 1000}}, "Anime-Planet rank <= 1000", ["title", "statistics.animeplanet.rank"]),

        # Kitsu Statistics (5 fields)
        ({"statistics.kitsu.score": {"gte": 7.0}}, "Kitsu score >= 7.0", ["title", "statistics.kitsu.score"]),
        ({"statistics.kitsu.members": {"gte": 1000}}, "Kitsu members >= 1000", ["title", "statistics.kitsu.members"]),
        ({"statistics.kitsu.favorites": {"gte": 10}}, "Kitsu favorites >= 10", ["title", "statistics.kitsu.favorites"]),
        ({"statistics.kitsu.rank": {"lte": 10000}}, "Kitsu rank <= 10000", ["title", "statistics.kitsu.rank"]),
        ({"statistics.kitsu.popularity_rank": {"lte": 10000}}, "Kitsu popularity_rank <= 10000", ["title", "statistics.kitsu.popularity_rank"]),

        # AnimeSchedule Statistics (4 fields)
        ({"statistics.animeschedule.score": {"gte": 6.0}}, "AnimeSchedule score >= 6.0", ["title", "statistics.animeschedule.score"]),
        ({"statistics.animeschedule.scored_by": {"gte": 5}}, "AnimeSchedule scored_by >= 5", ["title", "statistics.animeschedule.scored_by"]),
        ({"statistics.animeschedule.members": {"gte": 10}}, "AnimeSchedule members >= 10", ["title", "statistics.animeschedule.members"]),
        ({"statistics.animeschedule.rank": {"lte": 10000}}, "AnimeSchedule rank <= 10000", ["title", "statistics.animeschedule.rank"]),

        # Aggregate Score Field (1 field)
        ({"score.arithmetic_mean": {"gte": 7.0}}, "Aggregate score >= 7.0", ["title", "score.arithmetic_mean"]),
        ({"score.arithmetic_mean": {"gte": 6.0, "lte": 8.0}}, "Aggregate score 6.0-8.0", ["title", "score.arithmetic_mean"]),
    ]

    results = []

    print(f"Running {len(test_cases)} filter tests...\n")

    for filter_dict, description, expected_fields in test_cases:
        result = await test_filter(client, filter_dict, description, expected_fields)
        results.append(result)

        # Print result
        status_icon = "[PASS]" if "PASS" in result["status"] else "[FAIL]"
        print(f"{status_icon} {description}")
        print(f"   Filter: {filter_dict}")
        print(f"   Results: {result['result_count']} matches")

        if result["sample_results"]:
            print(f"   Samples:")
            for sample in result["sample_results"]:
                print(f"      - {sample['title']}: {sample['values']}")

        print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if "PASS" in r["status"])
    failed = sum(1 for r in results if "FAIL" in r["status"])

    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    # Group by platform
    platforms = {
        "MAL": [r for r in results if "MAL" in r["description"]],
        "AniList": [r for r in results if "AniList" in r["description"]],
        "AniDB": [r for r in results if "AniDB" in r["description"]],
        "Anime-Planet": [r for r in results if "Anime-Planet" in r["description"]],
        "Kitsu": [r for r in results if "Kitsu" in r["description"]],
        "AnimeSchedule": [r for r in results if "AnimeSchedule" in r["description"]],
        "Aggregate": [r for r in results if "Aggregate" in r["description"]],
    }

    for platform, platform_results in platforms.items():
        if platform_results:
            platform_passed = sum(1 for r in platform_results if "PASS" in r["status"])
            print(f"{platform}: {platform_passed}/{len(platform_results)} tests passed")

    print()

    # Test multi-platform combinations
    print("=" * 80)
    print("MULTI-PLATFORM COMBINATION TESTS")
    print("=" * 80)
    print()

    # Test 1: High score on multiple platforms
    print("Test: Anime with MAL score >= 7.0 AND AniList score >= 7.0")
    multi_filter_dict = {
        "statistics.mal.score": {"gte": 7.0},
        "statistics.anilist.score": {"gte": 7.0}
    }
    multi_filter = client._build_filter(multi_filter_dict)
    multi_results, _ = client.client.scroll(
        collection_name=client.collection_name,
        scroll_filter=multi_filter,
        limit=5,
        with_payload=True,
        with_vectors=False
    )

    print(f"Results: {len(multi_results)} matches")
    for point in multi_results[:3]:
        mal_score = _extract_nested_value(point.payload, "statistics.mal.score")
        anilist_score = _extract_nested_value(point.payload, "statistics.anilist.score")
        print(f"   - {point.payload['title']}: MAL={mal_score}, AniList={anilist_score}")
    print()

    # Test 2: Popular on MAL with high aggregate score
    print("Test: MAL members >= 50K AND aggregate score >= 7.0")
    combo_filter_dict = {
        "statistics.mal.members": {"gte": 50000},
        "score.arithmetic_mean": {"gte": 7.0}
    }
    combo_filter = client._build_filter(combo_filter_dict)
    combo_results, _ = client.client.scroll(
        collection_name=client.collection_name,
        scroll_filter=combo_filter,
        limit=5,
        with_payload=True,
        with_vectors=False
    )

    print(f"Results: {len(combo_results)} matches")
    for point in combo_results[:3]:
        members = _extract_nested_value(point.payload, "statistics.mal.members")
        avg_score = _extract_nested_value(point.payload, "score.arithmetic_mean")
        print(f"   - {point.payload['title']}: {members} members, {avg_score:.2f} score")
    print()

    print("=" * 80)
    print("COMPREHENSIVE TESTING COMPLETED")
    print("=" * 80)

    # Save detailed results
    output_path = Path(__file__).parent / "filter_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
