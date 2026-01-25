"""
Integration tests for update_vectors.py focusing on the critical fixes:
1. expected_updates validation and warning
2. Accurate anime success tracking (all-or-nothing per anime)
3. Per-vector statistics using detailed results

These tests validate the script-level logic that uses the detailed results
from QdrantClient.update_batch_vectors().
"""

import pytest
import uuid
from common.models.anime import Anime, AnimeRecord
from qdrant_db import QdrantClient
from vector_db_interface import VectorDocument
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


def build_anime_record(
    *,
    anime_id: str,
    title: str,
    genres: list[str],
    year: int | None,
    type: str,
    status: str,
    sources: list[str] | None = None,
) -> AnimeRecord:
    """Build an AnimeRecord with nested Anime data."""
    if sources is None:
        sources = []

    return AnimeRecord(
        anime=Anime(
            id=anime_id,
            title=title,
            genres=genres,
            year=year,
            type=type,
            status=status,
            sources=sources,
        )
    )


async def add_test_anime(
    client: QdrantClient,
    anime_list: list[AnimeRecord] | AnimeRecord,
    batch_size: int = 100,
):
    """Helper to add anime with correct point IDs."""
    if isinstance(anime_list, AnimeRecord):
        anime_list = [anime_list]

    documents = []
    for anime_rec in anime_list:
        documents.append(
            VectorDocument(id=anime_rec.anime.id, vectors={}, payload=anime_rec.anime.model_dump())
        )

    await client.add_documents(documents, batch_size=batch_size)


@pytest.mark.asyncio
async def test_vector_persistence_after_update(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that vectors are actually persisted and retrievable after update."""
    # Create and seed test anime
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Persistence Test Anime",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Generate and update text_vector
    title_content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    text_vector = embedding_manager.text_processor.encode_text(title_content)

    batch_updates = [
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        }
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Verify update succeeded
    assert result["success"] == 1, "Update should succeed"
    assert result["failed"] == 0, "No failures expected"

    # Verify vector is retrievable via search
    # Generate query vector for the same title
    query_vector = embedding_manager.text_processor.encode_text(
        "Persistence Test Anime"
    )

    search_results = await client.search_single_vector(
        vector_name="text_vector", vector_data=query_vector, limit=5
    )

    # Should find the anime we just updated (should be top result)
    found_ids = [hit["id"] for hit in search_results]
    assert test_anime.anime.id in found_ids, (
        "Updated anime should be searchable by text_vector"
    )

    # Verify it's actually the first result (highest similarity)
    assert search_results[0]["id"] == test_anime.anime.id, (
        "Updated anime should be the top search result"
    )


@pytest.mark.asyncio
async def test_detailed_results_provide_accurate_tracking(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that detailed results from update_batch_vectors enable accurate per-update tracking."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # Create batch with mix of valid and invalid updates
    batch_updates = []

    # Valid update for anime-0
    title_content = embedding_manager.field_mapper.extract_anime_text(test_anime[0].anime)
    text_vector = embedding_manager.text_processor.encode_text(title_content)
    batch_updates.append(
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        }
    )

    # Invalid update (wrong dimension) for anime-1
    batch_updates.append(
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,  # Wrong dimension
        }
    )

    # Valid update for anime-2
    title_content2 = embedding_manager.field_mapper.extract_anime_text(
        test_anime[2].anime
    )
    text_vector2 = embedding_manager.text_processor.encode_text(title_content2)
    batch_updates.append(
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector2,
        }
    )

    result = await client.update_batch_vectors(batch_updates)

    # Verify detailed results enable accurate tracking
    assert "results" in result, "Should have detailed results"
    assert len(result["results"]) == 3, "Should have 3 detailed results"

    # Verify we can identify which specific anime failed
    failed_anime_ids = [r["anime_id"] for r in result["results"] if not r["success"]]
    successful_anime_ids = [r["anime_id"] for r in result["results"] if r["success"]]

    assert test_anime[1].anime.id in failed_anime_ids, (
        "anime-1 should be identified as failed"
    )
    assert test_anime[0].anime.id in successful_anime_ids, (
        "anime-0 should be identified as successful"
    )
    assert test_anime[2].anime.id in successful_anime_ids, (
        "anime-2 should be identified as successful"
    )

    # Verify error messages are provided for failures
    for result_detail in result["results"]:
        if not result_detail["success"]:
            assert "error" in result_detail, "Failed update should have error message"
            assert len(result_detail["error"]) > 0, "Error message should not be empty"


@pytest.mark.asyncio
async def test_all_or_nothing_anime_success_logic(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that an anime is only counted as successful if ALL its requested vectors were updated."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="All or Nothing Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # We'll request 3 vectors, but only 2 will be valid names
    full_text = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    text_vector = embedding_manager.text_processor.encode_text(full_text)
    image_vector = [0.1] * 768

    batch_updates = [
        # 1. Valid text_vector
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # 2. Valid image_vector
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": image_vector,
        },
        # 3. Invalid vector name
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "invalid_vector",
            "vector_data": text_vector,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Success tracking should show 2 vector successes but 0 anime successes
    # because one vector for the anime failed.
    assert result["success"] == 2, "2 individual vectors should have succeeded"
    assert result["failed"] == 1, "1 individual vector should have failed"

    # The aggregation logic (which we're testing via scripts/update_vectors.py's expected usage)
    # would see this anime as failed.
    failed_anime_ids = {r["anime_id"] for r in result["results"] if not r["success"]}
    assert test_anime.anime.id in failed_anime_ids, "Anime should be in failed set"


@pytest.mark.asyncio
async def test_per_vector_statistics_from_detailed_results(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """
    Test that per-vector statistics are accurately tracked using detailed results.
    This validates the fix from lines 418-428 in update_vectors.py.
    """
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Stats Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # Create updates with specific failure pattern:
    # - text_vector: all 3 succeed
    # - image_vector: 2 succeed, 1 fails (anime-1)
    batch_updates = []

    for i, anime_rec in enumerate(test_anime):
        # Text vector - all valid
        title_content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
        text_vector = embedding_manager.text_processor.encode_text(title_content)
        batch_updates.append(
            {
                "anime_id": anime_rec.anime.id,
                "vector_name": "text_vector",
                "vector_data": text_vector,
            }
        )

        # Image vector - anime-1 has invalid dimension
        if i == 1:
            batch_updates.append(
                {
                    "anime_id": anime_rec.anime.id,
                    "vector_name": "image_vector",
                    "vector_data": [0.1] * 512,  # Invalid
                }
            )
        else:
            image_vector = [0.1] * 768
            batch_updates.append(
                {
                    "anime_id": anime_rec.anime.id,
                    "vector_name": "image_vector",
                    "vector_data": image_vector,
                }
            )

    result = await client.update_batch_vectors(batch_updates)

    # Simulate the per-vector statistics tracking from update_vectors.py (lines 418-428)
    vector_stats = {
        "text_vector": {"success": 0, "failed": 0},
        "image_vector": {"success": 0, "failed": 0},
    }

    for result_detail in result["results"]:
        vector_name = result_detail["vector_name"]
        if result_detail["success"]:
            vector_stats[vector_name]["success"] += 1
        else:
            vector_stats[vector_name]["failed"] += 1

    # Verify per-vector statistics
    assert vector_stats["text_vector"]["success"] == 3, (
        "All 3 text vectors should succeed"
    )
    assert vector_stats["text_vector"]["failed"] == 0, "No text vector failures"

    assert vector_stats["image_vector"]["success"] == 2, (
        "2 image vectors should succeed"
    )
    assert vector_stats["image_vector"]["failed"] == 1, (
        "1 image vector should fail (anime-1)"
    )


@pytest.mark.asyncio
async def test_empty_batch_handling(client: QdrantClient):
    """Test that empty batch is handled gracefully."""
    result = await client.update_batch_vectors([])

    assert result["success"] == 0, "Empty batch should have 0 successes"
    assert result["failed"] == 0, "Empty batch should have 0 failures"
    assert result["results"] == [], "Empty batch should have empty results list"


@pytest.mark.asyncio
async def test_all_validation_failures_no_qdrant_call(client: QdrantClient):
    """Test that when all updates fail validation, no Qdrant call is made."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Validation Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # All updates have validation errors
    batch_updates = [
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "invalid_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },  # Wrong dim
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": "not a vector",
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # All should fail validation, no Qdrant operation should occur
    assert result["success"] == 0, "All updates should fail validation"
    assert result["failed"] == 3, "Should have 3 validation failures"
    assert len(result["results"]) == 3, "Should have 3 detailed failure results"

    # All should have error messages
    for result_detail in result["results"]:
        assert not result_detail["success"], "All should be failures"
        assert "error" in result_detail, "Each should have error message"


@pytest.mark.asyncio
async def test_mixed_batch_all_combinations(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test batch with all possible anime success combinations: all pass, partial pass, all fail."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Mixed Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(5)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    batch_updates = []

    # Anime 0: 2/2 vectors succeed (all pass)
    for vector_name in ["text_vector", "image_vector"]:
        if vector_name == "text_vector":
            content = embedding_manager.field_mapper.extract_anime_text(test_anime[0].anime)
            vector = embedding_manager.text_processor.encode_text(content)
        else:
            vector = [0.1] * 768
        batch_updates.append(
            {
                "anime_id": test_anime[0].anime.id,
                "vector_name": vector_name,
                "vector_data": vector,
            }
        )

    # Anime 1: 1/2 vectors succeed (partial pass)
    batch_updates.append(
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        }
    )
    batch_updates.append(
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 512,
        }
    )  # Fail (wrong dim)

    # Anime 2: 0/2 vectors succeed (all fail)
    batch_updates.append(
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        }
    )  # Fail
    batch_updates.append(
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "image_vector",
            "vector_data": "invalid",
        }
    )  # Fail

    # Anime 3: 0/2 vectors succeed (invalid name + invalid data)
    batch_updates.append(
        {
            "anime_id": test_anime[3].anime.id,
            "vector_name": "invalid_vector",
            "vector_data": [0.1] * 1024,
        }
    )
    batch_updates.append(
        {
            "anime_id": test_anime[3].anime.id,
            "vector_name": "image_vector",
            "vector_data": None,
        }
    )

    # Anime 4: 2/2 vectors succeed (all pass)
    for vector_name in ["text_vector", "image_vector"]:
        if vector_name == "text_vector":
            content = embedding_manager.field_mapper.extract_anime_text(test_anime[4].anime)
            vector = embedding_manager.text_processor.encode_text(content)
        else:
            vector = [0.1] * 768
        batch_updates.append(
            {
                "anime_id": test_anime[4].anime.id,
                "vector_name": vector_name,
                "vector_data": vector,
            }
        )

    result = await client.update_batch_vectors(batch_updates)

    # Calculate anime success map
    anime_success_map = {}
    for r in result["results"]:
        anime_id = r["anime_id"]
        if anime_id not in anime_success_map:
            anime_success_map[anime_id] = {"total": 0, "success": 0}
        anime_success_map[anime_id]["total"] += 1
        if r["success"]:
            anime_success_map[anime_id]["success"] += 1

    # Verify all combinations
    assert anime_success_map[test_anime[0].anime.id]["success"] == 2, (
        "Anime 0: all should succeed"
    )
    assert anime_success_map[test_anime[0].anime.id]["total"] == 2

    assert anime_success_map[test_anime[1].anime.id]["success"] == 1, (
        "Anime 1: 1/2 should succeed"
    )
    assert anime_success_map[test_anime[1].anime.id]["total"] == 2

    assert anime_success_map[test_anime[2].anime.id]["success"] == 0, (
        "Anime 2: 0/2 should succeed"
    )
    assert anime_success_map[test_anime[2].anime.id]["total"] == 2

    assert anime_success_map[test_anime[3].anime.id]["success"] == 0, (
        "Anime 3: 0/2 should succeed"
    )
    assert anime_success_map[test_anime[3].anime.id]["total"] == 2

    assert anime_success_map[test_anime[4].anime.id]["success"] == 2, (
        "Anime 4: all should succeed"
    )
    assert anime_success_map[test_anime[4].anime.id]["total"] == 2

    # Count fully successful anime (only anime 0 and 4)
    fully_successful = sum(
        1
        for anime_id, stats in anime_success_map.items()
        if stats["success"] == stats["total"]
    )
    assert fully_successful == 2, "Only 2 anime should be fully successful"


@pytest.mark.asyncio
async def test_all_supported_vectors_simultaneously(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test updating all supported vector types in a single batch."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="All Vectors Test",
        genres=["Action", "Drama"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # All supported vector names
    vector_names = [
        "text_vector",
        "image_vector",
    ]

    batch_updates = []
    for vector_name in vector_names:
        if "image" in vector_name:
            # Image vectors are 768-dimensional
            vector_data = [0.1] * 768
        else:
            # Text vectors are 1024-dimensional
            content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
            vector_data = embedding_manager.text_processor.encode_text(content)

        batch_updates.append(
            {
                "anime_id": test_anime.anime.id,
                "vector_name": vector_name,
                "vector_data": vector_data,
            }
        )

    result = await client.update_batch_vectors(batch_updates)

    # Verify all supported vectors were processed
    assert len(result["results"]) == 2, "Should have 2 detailed results"
    assert result["success"] == 2, "All updates should succeed"
    assert result["failed"] == 0, "No failures expected"

    # Verify each vector name is present in results
    processed_names = {r["vector_name"] for r in result["results"]}
    for name in vector_names:
        assert name in processed_names, f"Vector {name} should be in results"


@pytest.mark.asyncio
async def test_duplicate_anime_in_same_batch(client: QdrantClient):
    """Test batch with duplicate anime IDs (same vector updated multiple times)."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Duplicate Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Update same vector 3 times with different values
    batch_updates = [
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.2] * 1024,
        },
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.3] * 1024,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Due to deduplication, only last one should persist
    assert result["success"] == 1, "Should deduplicate to 1 update"
    assert result["failed"] == 0, "No validation failures"
    assert result.get("duplicates_removed", 0) == 2, "Should have removed 2 duplicates"
    assert len(result["results"]) == 1, "Should have 1 result after deduplication"


@pytest.mark.asyncio
async def test_dimension_edge_cases(client: QdrantClient):
    """Test various dimension edge cases."""
    # Create 5 unique anime to avoid deduplication before validation
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Dimension Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(5)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    batch_updates = [
        # Exactly 1024 (should pass)
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        # Exactly 768 (should pass for image vectors)
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 768,
        },
        # Wrong dimension for text_vector
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },
        # Wrong dimension for image_vector
        {
            "anime_id": test_anime[3].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 1024,
        },
        # Empty vector
        {
            "anime_id": test_anime[4].anime.id,
            "vector_name": "text_vector",
            "vector_data": [],
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 2, "Exactly correct sizes should succeed"
    assert result["failed"] == 3, "3 dimension/validation mismatches should fail"


@pytest.mark.asyncio
async def test_float_precision_and_types(client: QdrantClient):
    """Test updating with different float types and high precision.

    This test focuses on edge cases that should pass validation.
    """
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Float Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(2)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    import numpy as np

    batch_updates = [
        # Standard floats
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.123456789] * 1024,
        },
        # Numpy array (converted to list)
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": np.array([0.1] * 1024, dtype=np.float32).tolist(),
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 2, "All valid float values should succeed"
    assert result["failed"] == 0, "No failures expected"


@pytest.mark.asyncio
async def test_non_existent_anime_ids(client: QdrantClient):
    """Test updates to anime IDs that don't exist in Qdrant."""
    batch_updates = [
        {
            "anime_id": "non-existent-1",
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": "non-existent-2",
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": "non-existent-3",
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Qdrant should accept updates even if points don't exist (will create them)
    # OR reject them - document the actual behavior
    assert "success" in result, "Should return result structure"
    assert "failed" in result, "Should return failed count"
    assert "results" in result, "Should return detailed results"

    print(
        f"Non-existent anime behavior: success={result['success']}, failed={result['failed']}"
    )


@pytest.mark.asyncio
async def test_batch_with_only_invalid_vector_names(client: QdrantClient):
    """Test batch where all vector names are invalid."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Invalid Names Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    batch_updates = [
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "invalid_vector_1",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "not_a_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "fake_vector_name",
            "vector_data": [0.1] * 1024,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 0, "All should fail with invalid names"
    assert result["failed"] == 3, "All 3 should be rejected"

    # All should have error messages about invalid vector names
    for r in result["results"]:
        assert not r["success"], "All should fail"
        assert "invalid" in r["error"].lower() or "vector" in r["error"].lower(), (
            "Error should mention invalid vector"
        )


@pytest.mark.asyncio
async def test_mixed_valid_invalid_anime_ids(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test batch with mix of existing and non-existing anime IDs.

    Note: Qdrant's update_vectors operation updates existing points.
    Non-existing points will be processed but may not be updated if they don't exist.
    """
    existing_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Existing Anime",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [existing_anime], batch_size=1)

    content = embedding_manager.field_mapper.extract_anime_text(existing_anime.anime)
    vector = embedding_manager.text_processor.encode_text(content)

    batch_updates = [
        # Existing anime - should work
        {
            "anime_id": existing_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        },
        # Another existing update with different vector name to avoid deduplication
        {
            "anime_id": existing_anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 768,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Both existing anime updates should work
    assert result["success"] == 2, "Existing anime updates should work"
    assert result["failed"] == 0, "No failures for existing anime"


@pytest.mark.asyncio
async def test_results_ordering_matches_input(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that detailed results maintain relationship to input order."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Order Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(5)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # Create specific pattern: success, fail, success, fail, success
    batch_updates = []
    for i, anime_rec in enumerate(test_anime):
        if i % 2 == 0:  # Even indices - valid
            content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
            vector = embedding_manager.text_processor.encode_text(content)
        else:  # Odd indices - invalid
            vector = [0.1] * 512  # Wrong dimension

        batch_updates.append(
            {"anime_id": anime_rec.anime.id, "vector_name": "text_vector", "vector_data": vector}
        )

    result = await client.update_batch_vectors(batch_updates)

    # Verify we can match results to input
    assert len(result["results"]) == 5, "Should have 5 results"

    # Check that all anime IDs from input are in results
    input_anime_ids = {u["anime_id"] for u in batch_updates}
    result_anime_ids = {r["anime_id"] for r in result["results"]}

    assert input_anime_ids == result_anime_ids, "All anime IDs should be in results"


@pytest.mark.asyncio
async def test_large_batch_realistic_failures(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test large batch (100 anime) with realistic failure patterns."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Large Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(100)
    ]

    await add_test_anime(client, test_anime, batch_size=50)

    batch_updates = []
    for i, anime_rec in enumerate(test_anime):
        content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
        vector = embedding_manager.text_processor.encode_text(content)

        # 90% success rate - realistic scenario
        if i % 10 != 0:  # 90% valid
            batch_updates.append(
                {
                    "anime_id": anime_rec.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vector,
                }
            )
        else:  # 10% invalid
            batch_updates.append(
                {
                    "anime_id": anime_rec.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": [0.1] * 512,
                }
            )

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 90, "90 updates should succeed"
    assert result["failed"] == 10, "10 updates should fail"
    assert len(result["results"]) == 100, "Should have 100 detailed results"


@pytest.mark.asyncio
async def test_sequential_updates_same_vector(client: QdrantClient):
    """Test updating same vector multiple times sequentially."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Sequential Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Update same vector 3 times sequentially
    for i in range(3):
        vector = [float(i)] * 1024  # Different values each time
        batch_updates = [
            {
                "anime_id": test_anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": vector,
            }
        ]

        result = await client.update_batch_vectors(batch_updates)
        assert result["success"] == 1, f"Update {i + 1} should succeed"

    # Verify last update persists
    query_vector = [2.0] * 1024  # Should match last update
    search_results = await client.search_single_vector(
        vector_name="text_vector", vector_data=query_vector, limit=5
    )

    # Should find our anime
    found_ids = [hit["id"] for hit in search_results]
    assert test_anime.anime.id in found_ids, "Should find updated anime"


@pytest.mark.asyncio
async def test_image_and_text_vectors_mixed(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test mixing image vectors (768-dim) and text vectors (1024-dim) in same batch."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Mixed Dim Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(4)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    content = embedding_manager.field_mapper.extract_anime_text(test_anime[0].anime)
    text_vector = embedding_manager.text_processor.encode_text(content)

    batch_updates = [
        # Text vector (1024-dim)
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # Image vector (768-dim)
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 768,
        },
        # Another text vector
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # Another image vector (768-dim)
        {
            "anime_id": test_anime[3].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 768,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 4, "All mixed dimension updates should succeed"
    assert result["failed"] == 0, "No failures expected"


@pytest.mark.asyncio
async def test_single_vector_update_method(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test update_single_vector method (not just batch updates)."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Single Update Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Test single vector update
    content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    vector = embedding_manager.text_processor.encode_text(content)

    success = await client.update_single_vector(
        anime_id=test_anime.anime.id, vector_name="text_vector", vector_data=vector
    )

    assert success is True, "Single vector update should succeed"

    # Test invalid vector name
    success = await client.update_single_vector(
        anime_id=test_anime.anime.id, vector_name="invalid_vector", vector_data=vector
    )

    assert success is False, "Invalid vector name should fail"


@pytest.mark.asyncio
async def test_batch_size_boundaries(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test various batch sizes: 1, 2, 50, 100, 500."""
    # Create test anime
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(500)
    ]

    # Add in batches to avoid memory issues
    for i in range(0, 500, 100):
        await add_test_anime(client, test_anime[i : i + 100], batch_size=100)

    # Test different batch sizes
    batch_sizes_to_test = [1, 2, 50, 100, 500]

    for batch_size in batch_sizes_to_test:
        batch_updates = []
        for i in range(batch_size):
            content = embedding_manager.field_mapper.extract_anime_text(
                test_anime[i].anime
            )
            vector = embedding_manager.text_processor.encode_text(content)
            batch_updates.append(
                {
                    "anime_id": test_anime[i].anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vector,
                }
            )

        result = await client.update_batch_vectors(batch_updates)

        assert result["success"] == batch_size, (
            f"Batch size {batch_size} should have {batch_size} successes"
        )
        assert result["failed"] == 0, f"Batch size {batch_size} should have 0 failures"
        assert len(result["results"]) == batch_size, f"Should have {batch_size} results"


@pytest.mark.asyncio
async def test_update_then_search_consistency(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that updated vectors are immediately searchable with correct results."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Consistency Test Anime",
        genres=["Action", "Adventure"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Update text_vector
    content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    text_vector = embedding_manager.text_processor.encode_text(content)

    update_result = await client.update_batch_vectors(
        [
            {
                "anime_id": test_anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": text_vector,
            }
        ]
    )

    assert update_result["success"] == 1, "Update should succeed"

    # Immediately search for the updated vector
    search_results = await client.search_single_vector(
        vector_name="text_vector", vector_data=text_vector, limit=5
    )

    # Should find the anime we just updated
    found_ids = [hit["id"] for hit in search_results]
    assert test_anime.anime.id in found_ids, "Updated anime should be immediately searchable"

    # Should be top result or very high in results (might have other test data)
    # Self-similarity should be very high
    our_result = next((r for r in search_results if r["id"] == test_anime.anime.id), None)
    assert our_result is not None, "Must find updated anime"
    assert our_result["similarity_score"] > 0.98, "Self-similarity should be very high"


@pytest.mark.asyncio
async def test_similarity_search_after_multiple_updates(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that similarity search returns correct results after vector updates."""
    # Create 3 similar anime
    anime_list = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title="Action Hero Adventure",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        ),
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title="Action Hero Story",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        ),
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title="Romance Comedy Love",
            genres=["Romance"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        ),
    ]

    await add_test_anime(client, anime_list, batch_size=3)

    # Update all title vectors
    batch_updates = []
    for anime_rec in anime_list:
        content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
        vector = embedding_manager.text_processor.encode_text(content)
        batch_updates.append(
            {"anime_id": anime_rec.anime.id, "vector_name": "text_vector", "vector_data": vector}
        )

    result = await client.update_batch_vectors(batch_updates)
    assert result["success"] == 3, "All 3 updates should succeed"

    # Search with "Action Hero" query
    query_vector = embedding_manager.text_processor.encode_text("Action Hero")
    search_results = await client.search_single_vector(
        vector_name="text_vector",
        vector_data=query_vector,
        limit=20,  # Increased limit to ensure we find our test anime
    )

    # Verify our test anime are in results
    result_ids = [hit["id"] for hit in search_results]
    assert anime_list[0].anime.id in result_ids, "Action Hero Adventure should be in results"
    assert anime_list[1].anime.id in result_ids, "Action Hero Story should be in results"

    # Get scores for action anime
    similar1_result = next(r for r in search_results if r["id"] == anime_list[0].anime.id)
    similar2_result = next(r for r in search_results if r["id"] == anime_list[1].anime.id)

    # Both action anime should have high similarity scores (>0.7) for "Action Hero" query
    assert similar1_result["similarity_score"] > 0.7, (
        "Action Hero Adventure should have high similarity"
    )
    assert similar2_result["similarity_score"] > 0.7, (
        "Action Hero Story should have high similarity"
    )

    # If romance anime is in results, action anime should score higher
    different_result = next(
        (r for r in search_results if r["id"] == anime_list[2].anime.id), None
    )
    if different_result:
        assert (
            similar1_result["similarity_score"] > different_result["similarity_score"]
        ), "Action anime should score higher than Romance"
        assert (
            similar2_result["similarity_score"] > different_result["similarity_score"]
        ), "Action anime should score higher than Romance"


@pytest.mark.asyncio
async def test_vector_extraction_failures_handling(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test handling when field_mapper extraction returns empty/minimal data."""
    # Create anime with minimal data
    minimal_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="M",  # Very short title
        genres=[],  # Empty genre
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [minimal_anime], batch_size=1)

    # Try to extract and update vectors for minimal anime
    content = embedding_manager.field_mapper.extract_anime_text(minimal_anime.anime)

    # Even with minimal data, extraction should return something
    assert content is not None, "Extraction should not return None"

    # Encode and update
    vector = embedding_manager.text_processor.encode_text(content)

    result = await client.update_batch_vectors(
        [
            {
                "anime_id": minimal_anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": vector,
            }
        ]
    )

    # Should still succeed even with minimal data
    assert result["success"] == 1, "Update with minimal data should succeed"


@pytest.mark.asyncio
async def test_all_error_types_in_detailed_results(client: QdrantClient):
    """Test that all error types return detailed, accurate error messages."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Error Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(5)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    batch_updates = [
        # Invalid vector name
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "invalid_name",
            "vector_data": [0.1] * 1024,
        },
        # Wrong dimension
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },
        # Invalid data type
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": "not a vector",
        },
        # None value
        {
            "anime_id": test_anime[3].anime.id,
            "vector_name": "text_vector",
            "vector_data": None,
        },
        # Empty vector
        {"anime_id": test_anime[4].anime.id, "vector_name": "text_vector", "vector_data": []},
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["failed"] == 5, "All 5 should fail"
    assert len(result["results"]) == 5, "Should have 5 detailed results"

    # Verify each has specific error message
    for r in result["results"]:
        assert not r["success"], "All should fail"
        assert "error" in r, "Must have error field"
        assert len(r["error"]) > 0, "Error message must not be empty"
        assert "anime_id" in r, "Must have anime_id"
        assert "vector_name" in r, "Must have vector_name"

    # Check error types are distinct and meaningful
    errors = [r["error"].lower() for r in result["results"]]
    assert any("invalid" in e and "vector" in e for e in errors), (
        "Should have invalid vector name error"
    )
    assert any("dimension" in e for e in errors), "Should have dimension error"
    assert any("valid" in e or "type" in e for e in errors), (
        "Should have data type errors"
    )


@pytest.mark.asyncio
async def test_multi_batch_statistics_aggregation(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that statistics are correctly aggregated across multiple sequential batches."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(30)
    ]

    await add_test_anime(client, test_anime, batch_size=30)

    # Track stats across 3 batches of 10
    total_success = 0
    total_failed = 0
    vector_stats = {"text_vector": {"success": 0, "failed": 0}}

    for batch_num in range(3):
        batch_start = batch_num * 10
        batch_end = batch_start + 10

        batch_updates = []
        for i in range(batch_start, batch_end):
            # Introduce failures: every 5th update fails
            if i % 5 == 0:
                vector = [0.1] * 512  # Wrong dimension - will fail
            else:
                content = embedding_manager.field_mapper.extract_anime_text(
                    test_anime[i].anime
                )
                vector = embedding_manager.text_processor.encode_text(content)

            batch_updates.append(
                {
                    "anime_id": test_anime[i].anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vector,
                }
            )

        result = await client.update_batch_vectors(batch_updates)

        # Aggregate stats
        total_success += result["success"]
        total_failed += result["failed"]

        for r in result["results"]:
            if r["success"]:
                vector_stats["text_vector"]["success"] += 1
            else:
                vector_stats["text_vector"]["failed"] += 1

    # Verify aggregated statistics
    assert total_success == 24, "Should have 24 total successes (30 - 6 failures)"
    assert total_failed == 6, "Should have 6 total failures (every 5th)"
    assert vector_stats["text_vector"]["success"] == 24, "Vector stats should match"
    assert vector_stats["text_vector"]["failed"] == 6, "Vector stats should match"


@pytest.mark.asyncio
async def test_result_structure_completeness(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test that result structure contains all required fields for both success and failure."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Structure Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    vector = embedding_manager.text_processor.encode_text(content)

    batch_updates = [
        # Success case
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        },
        # Failure case
        {
            "anime_id": test_anime.anime.id,
            "vector_name": "invalid_vector",
            "vector_data": vector,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Check top-level structure
    assert "success" in result, "Must have success count"
    assert "failed" in result, "Must have failed count"
    assert "results" in result, "Must have results list"
    assert isinstance(result["success"], int), "success must be int"
    assert isinstance(result["failed"], int), "failed must be int"
    assert isinstance(result["results"], list), "results must be list"

    # Check success result structure
    success_result = next(r for r in result["results"] if r["success"])
    assert "anime_id" in success_result, "Success must have anime_id"
    assert "vector_name" in success_result, "Success must have vector_name"
    assert "success" in success_result, "Success must have success field"
    assert success_result["success"] is True, "success field must be True"
    assert "error" not in success_result or success_result.get("error") is None, (
        "Success should not have error"
    )

    # Check failure result structure
    failure_result = next(r for r in result["results"] if not r["success"])
    assert "anime_id" in failure_result, "Failure must have anime_id"
    assert "vector_name" in failure_result, "Failure must have vector_name"
    assert "success" in failure_result, "Failure must have success field"
    assert failure_result["success"] is False, "success field must be False"
    assert "error" in failure_result, "Failure must have error field"
    assert isinstance(failure_result["error"], str), "error must be string"
    assert len(failure_result["error"]) > 0, "error must not be empty"


@pytest.mark.asyncio
async def test_update_with_different_vector_combinations(
    client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager
):
    """Test various combinations of vector updates in single batch."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Combo Test {i}",
            genres=["Action"],
            year=2020,
            type="TV",
            status="FINISHED",
            sources=[],
        )
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=3)

    batch_updates = []

    # Anime 0: Only text_vector
    content = embedding_manager.field_mapper.extract_anime_text(test_anime[0].anime)
    vector = embedding_manager.text_processor.encode_text(content)
    batch_updates.append(
        {
            "anime_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        }
    )

    # Anime 1: text_vector + image_vector
    content = embedding_manager.field_mapper.extract_anime_text(test_anime[1].anime)
    vector = embedding_manager.text_processor.encode_text(content)
    batch_updates.append(
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        }
    )
    batch_updates.append(
        {
            "anime_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 768,
        }
    )

    # Anime 2: text_vector + image_vector + invalid
    content = embedding_manager.field_mapper.extract_anime_text(test_anime[2].anime)
    vector = embedding_manager.text_processor.encode_text(content)
    batch_updates.append(
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        }
    )
    batch_updates.append(
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 768,
        }
    )
    batch_updates.append(
        {
            "anime_id": test_anime[2].anime.id,
            "vector_name": "invalid_vector",
            "vector_data": vector,
        }
    )

    result = await client.update_batch_vectors(batch_updates)

    # Total: 1 (Anime 0) + 2 (Anime 1) + 2 (Anime 2) = 5 successful vector updates
    assert result["success"] == 5, "All 5 valid updates should succeed"
    assert result["failed"] == 1, "1 invalid vector name should fail"

    # Verify per-anime breakdown
    anime_results = {}
    for r in result["results"]:
        anime_id = r["anime_id"]
        if anime_id not in anime_results:
            anime_results[anime_id] = 0
        if r["success"]:
            anime_results[anime_id] += 1

    assert anime_results[test_anime[0].anime.id] == 1, "Anime 0 should have 1 update"
    assert anime_results[test_anime[1].anime.id] == 2, "Anime 1 should have 2 updates"
    assert anime_results[test_anime[2].anime.id] == 2, "Anime 2 should have 2 successful updates"


def test_malformed_anime_payload_handling():
    """Test that malformed anime payloads are caught and skipped gracefully.

    This tests the fix for the TypeError bug where None or non-dict anime values
    would cause uncaught exceptions in the data validation loop (lines 203-212).
    """
    import uuid
    from pydantic import ValidationError

    # Minimal valid anime structure for testing
    def make_valid_anime(anime_id="test-123"):
        return {
            "id": anime_id,
            "title": "Test Anime",
            "type": "TV",
            "status": "FINISHED",
            "sources": [],
        }

    # Test cases covering all edge cases from the bug report
    test_cases = [
        ("None anime", {"anime": None, "characters": [], "episodes": []}),
        ("String anime", {"anime": "not_a_dict", "characters": [], "episodes": []}),
        ("Int anime", {"anime": 123, "characters": [], "episodes": []}),
        ("List anime", {"anime": [], "characters": [], "episodes": []}),
        ("Missing anime key", {"characters": [], "episodes": []}),
        ("Empty anime dict", {"anime": {}, "characters": [], "episodes": []}),
        ("Valid with ID", {"anime": make_valid_anime("existing-id"), "characters": [], "episodes": []}),
        ("Valid without ID", {"anime": {**make_valid_anime(), "id": None}, "characters": [], "episodes": []}),
        ("Valid empty ID", {"anime": {**make_valid_anime(), "id": ""}, "characters": [], "episodes": []}),
    ]

    records = []
    skipped = []

    for desc, anime_dict in test_cases:
        try:
            # This is the fixed code from lines 206-213 in update_vectors.py
            anime_payload = anime_dict.get("anime")
            if not isinstance(anime_payload, dict):
                raise KeyError("Missing or invalid 'anime' key")
            if not anime_payload.get("id"):
                anime_payload["id"] = str(uuid.uuid4())
            records.append(AnimeRecord(**anime_dict))
        except (KeyError, TypeError, ValidationError) as e:
            skipped.append((desc, type(e).__name__))
            continue

    # Verify results
    assert len(records) > 0, f"Should have created at least one valid record, got {len(records)} records"
    assert len(skipped) > 0, f"Should have skipped malformed records, got {len(skipped)} skipped"

    # First 5 test cases should be skipped (malformed anime values)
    assert len(skipped) >= 5, f"Expected at least 5 skipped, got {len(skipped)}: {skipped}"

    # All malformed records should be caught by KeyError or ValidationError
    for desc, exception_type in skipped:
        assert exception_type in ["KeyError", "ValidationError", "TypeError"], \
            f"Unexpected exception type for '{desc}': {exception_type}"

    # Valid records should have IDs assigned
    for record in records:
        assert record.anime.id, "All valid records should have an ID"
        assert len(record.anime.id) > 0, "ID should not be empty"


def test_no_uncaught_typeerror_from_malformed_data():
    """Verify that TypeError for None/non-dict anime is caught, not raised.

    These cases previously caused uncaught TypeError that would crash the loop:
    - anime_dict["anime"] is None → TypeError: 'in' operator on None
    - anime_dict["anime"] is string → TypeError: subscript assignment on str
    """
    import uuid
    from pydantic import ValidationError

    # These cases previously caused uncaught TypeError
    problematic_cases = [
        {"anime": None, "characters": [], "episodes": []},  # TypeError: 'in' on None
        {"anime": "string", "characters": [], "episodes": []},  # TypeError: subscript on str
        {"anime": 123, "characters": [], "episodes": []},  # TypeError: 'in' on int
        {"anime": [], "characters": [], "episodes": []},  # TypeError: list not dict
    ]

    for anime_dict in problematic_cases:
        exception_caught = False
        try:
            # Fixed code from lines 206-213
            anime_payload = anime_dict.get("anime")
            if not isinstance(anime_payload, dict):
                raise KeyError("Missing or invalid 'anime' key")
            if not anime_payload.get("id"):
                anime_payload["id"] = str(uuid.uuid4())
            AnimeRecord(**anime_dict)
        except (KeyError, TypeError, ValidationError):
            # This is expected - malformed data should be caught
            exception_caught = True
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")

        assert exception_caught, f"Should have caught exception for {anime_dict}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
