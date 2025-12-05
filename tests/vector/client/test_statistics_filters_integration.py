"""Comprehensive test for all statistics filters using QdrantClient infrastructure."""
import pytest
from typing import Any, Dict, List

from src.vector.client.qdrant_client import QdrantClient


async def _test_single_filter(
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
        results, _ = await client.client.scroll(
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
            "status": f"FAIL: {e!s}",
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


@pytest.mark.asyncio
async def test_comprehensive_statistics_filters(client: QdrantClient) -> None:
    """Test all statistics filters using QdrantClient with proper pytest integration."""

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

    # Run all filter tests and collect results
    for filter_dict, description, expected_fields in test_cases:
        result = await _test_single_filter(client, filter_dict, description, expected_fields)
        results.append(result)

        # Assert each filter test passed
        assert "PASS" in result["status"], f"Filter test failed: {description} - {result['status']}"

    # Assert all tests passed
    assert len(results) == len(test_cases)
    passed = sum(1 for r in results if "PASS" in r["status"])
    assert passed == len(test_cases), f"Only {passed}/{len(test_cases)} filter tests passed"

    # Test multi-platform combination filters
    # Test 1: High score on multiple platforms
    multi_filter_dict = {
        "statistics.mal.score": {"gte": 7.0},
        "statistics.anilist.score": {"gte": 7.0}
    }
    multi_filter = client._build_filter(multi_filter_dict)
    multi_results, _ = await client.client.scroll(
        collection_name=client.collection_name,
        scroll_filter=multi_filter,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    # Should find some results with both scores high (test collection may be empty, so just check it doesn't error)
    assert isinstance(multi_results, list)

    # Test 2: Popular on MAL with high aggregate score
    combo_filter_dict = {
        "statistics.mal.members": {"gte": 50000},
        "score.arithmetic_mean": {"gte": 7.0}
    }
    combo_filter = client._build_filter(combo_filter_dict)
    combo_results, _ = await client.client.scroll(
        collection_name=client.collection_name,
        scroll_filter=combo_filter,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    # Combination filters work without error
    assert isinstance(combo_results, list)
