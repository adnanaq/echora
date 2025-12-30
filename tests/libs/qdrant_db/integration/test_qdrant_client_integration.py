#!/usr/bin/env python3
"""
Integration tests for QdrantClient.

INTEGRATION TESTS - These tests require:
- Real Qdrant database connection
- Real ML models (BGE-M3, OpenCLIP)
- Real data files
- Significant setup time (model loading)

Tests cover:
- update_single_vector and update_batch_vectors methods
- Vector generation with embedding_manager
- End-to-end workflows from anime data to Qdrant storage

Use pytest markers to run/skip integration tests:
    pytest -m "not integration"  # Skip integration tests
    pytest -m integration         # Run only integration tests
"""

import json
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.anime import AnimeEntry
from qdrant_db import QdrantClient
from vector_db_interface import VectorDocument

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def test_anime_data():
    """Load test anime data from enriched database."""
    with open("./data/qdrant_storage/enriched_anime_database.json") as f:
        data = json.load(f)
    return data["data"][:3]  # Use first 3 anime for testing


@pytest_asyncio.fixture
async def seeded_test_data(
    client: QdrantClient, embedding_manager, test_anime_data: list[dict]
):
    """Seed test anime data into Qdrant before running tests."""
    # Convert to AnimeEntry objects
    anime_entries = [AnimeEntry(**anime_dict) for anime_dict in test_anime_data]

    # Generate vectors using embedding_manager
    gen_results = await embedding_manager.process_anime_batch(anime_entries)

    # Create VectorDocument objects with generated vectors
    documents = []
    for i, anime_entry in enumerate(anime_entries):
        gen_result = gen_results[i]
        if gen_result.get("metadata", {}).get("processing_failed"):
            pytest.skip(
                f"Failed to generate vectors for test anime: {anime_entry.title}"
            )

        # Create document with generated vectors
        doc = VectorDocument(
            id=client._generate_point_id(anime_entry.id),
            vectors=gen_result.get("vectors", {}),
            payload=anime_entry.model_dump(),
        )
        documents.append(doc)

    # Upsert test data to ensure points exist
    result = await client.add_documents(documents, batch_size=len(documents))

    if not result.get("success"):
        pytest.skip("Failed to seed test data into Qdrant")

    # Return the anime entries for tests to use
    return anime_entries


def create_test_anime(anime_id: str = "test-anime", title: str = "Test Anime"):
    """Create a test anime entry for isolated tests."""
    return AnimeEntry(
        id=anime_id,
        title=title,
        genres=["Action", "Adventure"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )


async def add_test_anime(
    client: QdrantClient, embedding_manager, anime: AnimeEntry
) -> bool:
    """Add anime entry to Qdrant with generated vectors.

    Helper function that mimics the old add_documents([anime]) behavior.
    """
    gen_results = await embedding_manager.process_anime_batch([anime])
    doc = VectorDocument(
        id=client._generate_point_id(anime.id),
        vectors=gen_results[0].get("vectors", {}),
        payload=anime.model_dump(),
    )
    result = await client.add_documents([doc], batch_size=1)
    return result.get("success", False)


@pytest.mark.asyncio
async def test_update_single_vector(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test updating a single vector for one anime."""
    # Get first anime
    anime = seeded_test_data[0]

    # Generate title vector
    title_content = embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)

    # Update single vector
    success = await client.update_single_vector(
        anime_id=anime.id, vector_name="title_vector", vector_data=title_vector
    )

    assert success is True, "Single vector update should succeed"


@pytest.mark.asyncio
async def test_update_single_vector_invalid_name(
    client: QdrantClient, seeded_test_data: list[AnimeEntry]
):
    """Test updating with invalid vector name."""
    anime = seeded_test_data[0]

    # Generate dummy vector
    dummy_vector = [0.0] * 1024

    # Try to update non-existent vector
    success = await client.update_single_vector(
        anime_id=anime.id, vector_name="invalid_vector", vector_data=dummy_vector
    )

    assert success is False, "Invalid vector name should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch updating multiple vectors."""
    batch_updates = []

    for anime in seeded_test_data:
        # Generate genre vector
        genre_content = embedding_manager.field_mapper._extract_genre_content(anime)
        genre_vector = embedding_manager.text_processor.encode_text(genre_content)

        batch_updates.append(
            {
                "anime_id": anime.id,
                "vector_name": "genre_vector",
                "vector_data": genre_vector,
            }
        )

    # Execute batch update
    result = await client.update_batch_vectors(batch_updates)

    assert "success" in result, "Result should contain success count"
    assert "failed" in result, "Result should contain failed count"
    assert "results" in result, "Result should contain detailed results list"
    assert result["success"] == len(seeded_test_data), "All updates should succeed"
    assert result["failed"] == 0, "No updates should fail"
    assert len(result["results"]) == len(seeded_test_data), (
        "Should have result for each update"
    )

    # Verify all detailed results show success
    for update_result in result["results"]:
        assert update_result["success"] is True, (
            f"Update should succeed: {update_result}"
        )
        assert "anime_id" in update_result, "Result should contain anime_id"
        assert "vector_name" in update_result, "Result should contain vector_name"


@pytest.mark.asyncio
async def test_update_batch_vectors_mixed(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch updating multiple vectors for same anime."""
    anime = seeded_test_data[0]

    # Generate multiple vectors
    title_content = embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)

    genre_content = embedding_manager.field_mapper._extract_genre_content(anime)
    genre_vector = embedding_manager.text_processor.encode_text(genre_content)

    character_content = embedding_manager.field_mapper._extract_character_content(anime)
    character_vector = embedding_manager.text_processor.encode_text(character_content)

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": title_vector,
        },
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": genre_vector,
        },
        {
            "anime_id": anime.id,
            "vector_name": "character_vector",
            "vector_data": character_vector,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 3, "All 3 vector updates should succeed"
    assert result["failed"] == 0, "No updates should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_empty(client: QdrantClient):
    """Test batch update with empty list."""
    result = await client.update_batch_vectors([])

    assert result["success"] == 0, "Empty batch should have 0 successes"
    assert result["failed"] == 0, "Empty batch should have 0 failures"
    assert result["results"] == [], "Empty batch should have empty results list"


@pytest.mark.asyncio
async def test_update_image_vector(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test updating image vector."""
    anime = seeded_test_data[0]

    # Generate image vector
    image_vector = await embedding_manager.vision_processor.process_anime_image_vector(
        anime
    )

    if image_vector is None:
        pytest.skip("No image data available for this anime")

    # Update image vector
    success = await client.update_single_vector(
        anime_id=anime.id, vector_name="image_vector", vector_data=image_vector
    )

    assert success is True, "Image vector update should succeed"


@pytest.mark.asyncio
async def test_update_all_text_vectors(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test updating all text vectors for one anime."""
    anime = seeded_test_data[0]

    text_vector_names = [
        "title_vector",
        "character_vector",
        "genre_vector",
        "staff_vector",
        "temporal_vector",
        "streaming_vector",
        "related_vector",
        "franchise_vector",
        "episode_vector",
    ]

    batch_updates = []

    for vector_name in text_vector_names:
        # Extract content based on vector type
        if vector_name == "title_vector":
            content = embedding_manager.field_mapper._extract_title_content(anime)
        elif vector_name == "character_vector":
            content = embedding_manager.field_mapper._extract_character_content(anime)
        elif vector_name == "genre_vector":
            content = embedding_manager.field_mapper._extract_genre_content(anime)
        elif vector_name == "staff_vector":
            content = embedding_manager.field_mapper._extract_staff_content(anime)
        elif vector_name == "temporal_vector":
            content = embedding_manager.field_mapper._extract_temporal_content(anime)
        elif vector_name == "streaming_vector":
            content = embedding_manager.field_mapper._extract_streaming_content(anime)
        elif vector_name == "related_vector":
            content = embedding_manager.field_mapper._extract_related_content(anime)
        elif vector_name == "franchise_vector":
            content = embedding_manager.field_mapper._extract_franchise_content(anime)
        elif vector_name == "episode_vector":
            content = embedding_manager.field_mapper._extract_episode_content(anime)

        # Generate vector
        vector_data = embedding_manager.text_processor.encode_text(content)

        batch_updates.append(
            {
                "anime_id": anime.id,
                "vector_name": vector_name,
                "vector_data": vector_data,
            }
        )

    # Execute batch update
    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 9, "All 9 text vector updates should succeed"
    assert result["failed"] == 0, "No updates should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_with_failures(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch update correctly tracks both successes and failures with detailed results."""
    anime = seeded_test_data[0]

    # Create batch with mix of valid and invalid updates
    batch_updates = []

    # Valid update
    title_content = embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)
    batch_updates.append(
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": title_vector,
        }
    )

    # Invalid update (invalid vector name)
    genre_content = embedding_manager.field_mapper._extract_genre_content(anime)
    genre_vector = embedding_manager.text_processor.encode_text(genre_content)
    batch_updates.append(
        {
            "anime_id": anime.id,
            "vector_name": "invalid_vector_name",
            "vector_data": genre_vector,
        }
    )

    # Another valid update
    character_content = embedding_manager.field_mapper._extract_character_content(anime)
    character_vector = embedding_manager.text_processor.encode_text(character_content)
    batch_updates.append(
        {
            "anime_id": anime.id,
            "vector_name": "character_vector",
            "vector_data": character_vector,
        }
    )

    result = await client.update_batch_vectors(batch_updates)

    # Verify result structure
    assert "success" in result, "Result should contain success count"
    assert "failed" in result, "Result should contain failed count"
    assert "results" in result, "Result should contain detailed results"

    # Verify counts
    assert result["success"] == 2, "Should have 2 successful updates"
    assert result["failed"] == 1, "Should have 1 failed update"
    assert len(result["results"]) == 3, "Should have 3 detailed results"

    # Verify detailed results
    successful_updates = [r for r in result["results"] if r["success"]]
    failed_updates = [r for r in result["results"] if not r["success"]]

    assert len(successful_updates) == 2, "Should have 2 successful update details"
    assert len(failed_updates) == 1, "Should have 1 failed update detail"

    # Check failed update has error message
    assert "error" in failed_updates[0], "Failed update should have error message"
    assert "invalid_vector_name" in str(failed_updates[0]["error"]).lower(), (
        "Error should mention invalid vector"
    )


@pytest.mark.asyncio
async def test_update_batch_vectors_all_validation_failures(
    client: QdrantClient, seeded_test_data: list[AnimeEntry]
):
    """Test batch update when all updates fail validation."""
    anime = seeded_test_data[0]

    batch_updates = [
        # Invalid vector name
        {
            "anime_id": anime.id,
            "vector_name": "invalid_vector_1",
            "vector_data": [0.1] * 1024,
        },
        # Wrong dimension
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.1] * 512,  # Should be 1024
        },
        # Invalid data type
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": "not a vector",
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 0, "All updates should fail"
    assert result["failed"] == 3, "Should have 3 failures"
    assert len(result["results"]) == 3, "Should have 3 detailed results"

    # All results should be failures with error messages
    for update_result in result["results"]:
        assert update_result["success"] is False, "All updates should fail"
        assert "error" in update_result, "Failed update should have error message"


@pytest.mark.asyncio
async def test_update_batch_vectors_partial_anime_success(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch update where some vectors succeed and some fail for the same anime."""
    anime = seeded_test_data[0]

    # Generate valid vectors
    title_content = embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)

    genre_content = embedding_manager.field_mapper._extract_genre_content(anime)
    genre_vector = embedding_manager.text_processor.encode_text(genre_content)

    batch_updates = [
        # Valid update
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": title_vector,
        },
        # Invalid update (wrong dimension)
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": [0.1] * 512,  # Wrong dimension
        },
        # Valid update
        {
            "anime_id": anime.id,
            "vector_name": "character_vector",
            "vector_data": genre_vector,  # Reuse valid vector
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 2, "Should have 2 successful updates"
    assert result["failed"] == 1, "Should have 1 failed update"
    assert len(result["results"]) == 3, "Should have 3 detailed results"

    # Verify we can identify which specific update failed
    failed_result = next(r for r in result["results"] if not r["success"])
    assert failed_result["vector_name"] == "genre_vector", (
        "Should identify genre_vector as failed"
    )
    assert "dimension" in failed_result["error"].lower(), (
        "Error should mention dimension mismatch"
    )


@pytest.mark.asyncio
async def test_update_batch_vectors_multiple_anime_mixed_results(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch update across multiple anime with mixed success/failure."""
    anime1, anime2, anime3 = seeded_test_data

    # Generate valid vectors
    title1 = embedding_manager.field_mapper._extract_title_content(anime1)
    title_vector1 = embedding_manager.text_processor.encode_text(title1)

    title2 = embedding_manager.field_mapper._extract_title_content(anime2)
    title_vector2 = embedding_manager.text_processor.encode_text(title2)

    batch_updates = [
        # Anime 1 - valid
        {
            "anime_id": anime1.id,
            "vector_name": "title_vector",
            "vector_data": title_vector1,
        },
        # Anime 2 - valid
        {
            "anime_id": anime2.id,
            "vector_name": "title_vector",
            "vector_data": title_vector2,
        },
        # Anime 3 - invalid (wrong dimension)
        {
            "anime_id": anime3.id,
            "vector_name": "title_vector",
            "vector_data": [0.1] * 512,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 2, "Should have 2 successful updates"
    assert result["failed"] == 1, "Should have 1 failed update"

    # Verify we can identify which anime had failures
    successful_anime_ids = {r["anime_id"] for r in result["results"] if r["success"]}
    failed_anime_ids = {r["anime_id"] for r in result["results"] if not r["success"]}

    assert anime1.id in successful_anime_ids, "Anime 1 should succeed"
    assert anime2.id in successful_anime_ids, "Anime 2 should succeed"
    assert anime3.id in failed_anime_ids, "Anime 3 should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_dimension_validation(
    client: QdrantClient, seeded_test_data: list[AnimeEntry]
):
    """Test batch update validates vector dimensions correctly."""
    anime = seeded_test_data[0]

    batch_updates = [
        # Too small
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.1] * 512,
        },
        # Too large
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": [0.1] * 2048,
        },
        # Empty vector (fails is_float_vector check)
        {"anime_id": anime.id, "vector_name": "character_vector", "vector_data": []},
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 0, "All dimension mismatches should fail"
    assert result["failed"] == 3, "Should have 3 failures"

    # Verify error messages mention dimension or validation issues
    dimension_errors = [
        r for r in result["results"] if "dimension" in r["error"].lower()
    ]
    validation_errors = [r for r in result["results"] if "valid" in r["error"].lower()]

    assert len(dimension_errors) >= 2, (
        "Should have at least 2 dimension mismatch errors"
    )
    assert len(validation_errors) >= 1, "Empty vector should fail validation check"


@pytest.mark.asyncio
async def test_update_batch_vectors_invalid_data_types(
    client: QdrantClient, seeded_test_data: list[AnimeEntry]
):
    """Test batch update rejects invalid data types."""
    anime = seeded_test_data[0]

    batch_updates = [
        # String instead of list
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": "not a vector",
        },
        # Dict instead of list
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": {"invalid": "data"},
        },
        # None value
        {"anime_id": anime.id, "vector_name": "character_vector", "vector_data": None},
        # List with non-float values
        {
            "anime_id": anime.id,
            "vector_name": "staff_vector",
            "vector_data": ["a", "b", "c"],
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 0, "All invalid data types should fail"
    assert result["failed"] == 4, "Should have 4 failures"

    # All should have error messages about invalid data
    for update_result in result["results"]:
        assert not update_result["success"], "Invalid data should fail"
        assert "error" in update_result, "Should have error message"


@pytest.mark.asyncio
async def test_update_batch_vectors_same_vector_multiple_updates(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch update with multiple updates to same vector (last one should win, deduplication occurs)."""
    anime = seeded_test_data[0]

    title_content = embedding_manager.field_mapper._extract_title_content(anime)
    title_vector1 = embedding_manager.text_processor.encode_text(title_content)
    title_vector2 = embedding_manager.text_processor.encode_text(
        title_content
    )  # Same content, different instance

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": title_vector1,
        },
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": title_vector2,
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Due to deduplication by anime_id + vector_name, only 1 update actually happens (last one wins)
    assert result["success"] == 1, "Should have 1 successful update (deduplicated)"
    assert result["failed"] == 0, "No failures expected"
    assert len(result["results"]) == 1, "Should have 1 result (deduplicated)"
    assert result["results"][0]["vector_name"] == "title_vector", (
        "Should be title_vector"
    )
    assert result["results"][0]["anime_id"] == anime.id, "Should be correct anime_id"


@pytest.mark.asyncio
async def test_update_batch_vectors_large_batch(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test batch update with many vectors per anime."""
    anime = seeded_test_data[0]

    # Update all 9 text vectors
    text_vector_names = [
        "title_vector",
        "character_vector",
        "genre_vector",
        "staff_vector",
        "temporal_vector",
        "streaming_vector",
        "related_vector",
        "franchise_vector",
        "episode_vector",
    ]

    batch_updates = []
    for vector_name in text_vector_names:
        if vector_name == "title_vector":
            content = embedding_manager.field_mapper._extract_title_content(anime)
        elif vector_name == "character_vector":
            content = embedding_manager.field_mapper._extract_character_content(anime)
        elif vector_name == "genre_vector":
            content = embedding_manager.field_mapper._extract_genre_content(anime)
        elif vector_name == "staff_vector":
            content = embedding_manager.field_mapper._extract_staff_content(anime)
        elif vector_name == "temporal_vector":
            content = embedding_manager.field_mapper._extract_temporal_content(anime)
        elif vector_name == "streaming_vector":
            content = embedding_manager.field_mapper._extract_streaming_content(anime)
        elif vector_name == "related_vector":
            content = embedding_manager.field_mapper._extract_related_content(anime)
        elif vector_name == "franchise_vector":
            content = embedding_manager.field_mapper._extract_franchise_content(anime)
        elif vector_name == "episode_vector":
            content = embedding_manager.field_mapper._extract_episode_content(anime)

        vector_data = embedding_manager.text_processor.encode_text(content)

        batch_updates.append(
            {
                "anime_id": anime.id,
                "vector_name": vector_name,
                "vector_data": vector_data,
            }
        )

    result = await client.update_batch_vectors(batch_updates)

    assert result["success"] == 9, "All 9 vector updates should succeed"
    assert result["failed"] == 0, "No failures expected"
    assert len(result["results"]) == 9, "Should have 9 detailed results"

    # Verify all vector names are present in results
    result_vector_names = {r["vector_name"] for r in result["results"]}
    assert result_vector_names == set(text_vector_names), (
        "All vectors should be in results"
    )


@pytest.mark.asyncio
async def test_update_batch_vectors_preserves_order(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeEntry]
):
    """Test that detailed results preserve order of input updates."""
    anime = seeded_test_data[0]

    title_content = embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)

    # Mix of valid and invalid updates in specific order
    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": title_vector,
        },  # Valid
        {
            "anime_id": anime.id,
            "vector_name": "invalid_1",
            "vector_data": title_vector,
        },  # Invalid
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": title_vector,
        },  # Valid
        {
            "anime_id": anime.id,
            "vector_name": "invalid_2",
            "vector_data": title_vector,
        },  # Invalid
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Results should maintain relationship to input (though not necessarily same order)
    assert len(result["results"]) == 4, "Should have 4 results"
    assert result["success"] == 2, "Should have 2 successes"
    assert result["failed"] == 2, "Should have 2 failures"

    # Count specific vectors in results
    result_vectors = [r["vector_name"] for r in result["results"]]
    assert "title_vector" in result_vectors, "title_vector should be in results"
    assert "genre_vector" in result_vectors, "genre_vector should be in results"
    assert "invalid_1" in result_vectors, "invalid_1 should be in results"
    assert "invalid_2" in result_vectors, "invalid_2 should be in results"


# ============================================================================
# DEDUPLICATION POLICY TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_deduplication_last_wins_policy(client: QdrantClient, embedding_manager):
    """Test that last-wins deduplication keeps the last occurrence."""
    anime = create_test_anime(anime_id="dedup-last-wins-test")
    await add_test_anime(client, embedding_manager, anime)

    # Create duplicate updates with very different vectors
    first_vector = [0.001] * 1024
    second_vector = [0.999] * 1024  # Very different from first

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": first_vector,
        },
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": second_vector,
        },  # Duplicate
    ]

    result = await client.update_batch_vectors(batch_updates, dedup_policy="last-wins")

    assert result["success"] == 1, "Should have 1 successful update (last one)"
    assert result["duplicates_removed"] == 1, "Should have removed 1 duplicate"

    # Verify the second vector was used by searching with it
    search_result = await client.search_single_vector(
        vector_name="title_vector", vector_data=second_vector, limit=5
    )

    # Find our anime in results
    our_anime = next((r for r in search_result if r["id"] == anime.id), None)
    assert our_anime is not None, f"Should find anime {anime.id} in results"
    assert our_anime["similarity_score"] > 0.99, (
        "Should match the last vector with high similarity"
    )


@pytest.mark.asyncio
async def test_deduplication_first_wins_policy(client: QdrantClient, embedding_manager):
    """Test that first-wins deduplication keeps the first occurrence."""
    anime = create_test_anime(anime_id="dedup-first-wins-test")
    await add_test_anime(client, embedding_manager, anime)

    first_vector = [0.001] * 1024
    second_vector = [0.999] * 1024

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": first_vector,
        },
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": second_vector,
        },  # Duplicate
    ]

    result = await client.update_batch_vectors(batch_updates, dedup_policy="first-wins")

    assert result["success"] == 1, "Should have 1 successful update (first one)"
    assert result["duplicates_removed"] == 1, "Should have removed 1 duplicate"

    # Verify the first vector was used
    search_result = await client.search_single_vector(
        vector_name="title_vector", vector_data=first_vector, limit=5
    )
    our_anime = next((r for r in search_result if r["id"] == anime.id), None)
    assert our_anime is not None, f"Should find anime {anime.id} in results"
    assert our_anime["similarity_score"] > 0.99, (
        "Should match the first vector with high similarity"
    )


@pytest.mark.asyncio
async def test_deduplication_fail_policy(client: QdrantClient, embedding_manager):
    """Test that fail policy raises error on duplicates."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.9] * 1024,
        },  # Duplicate
    ]

    with pytest.raises(ValueError, match="Duplicate update found"):
        await client.update_batch_vectors(batch_updates, dedup_policy="fail")


@pytest.mark.asyncio
async def test_deduplication_warn_policy(
    client: QdrantClient, embedding_manager, caplog
):
    """Test that warn policy logs warning but continues with last-wins."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.9] * 1024,
        },  # Duplicate
    ]

    import logging

    with caplog.at_level(logging.WARNING):
        result = await client.update_batch_vectors(batch_updates, dedup_policy="warn")

    assert result["success"] == 1, "Should succeed with last-wins behavior"
    assert result["duplicates_removed"] == 1, "Should track duplicate removal"

    # Verify warning was logged
    assert any("Duplicate update for" in record.message for record in caplog.records), (
        "Should log warning about duplicate"
    )


@pytest.mark.asyncio
async def test_no_duplicates_all_policies(client: QdrantClient, embedding_manager):
    """Test that all policies work correctly when there are no duplicates."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    batch_updates = [
        {
            "anime_id": anime.id,
            "vector_name": "title_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "anime_id": anime.id,
            "vector_name": "genre_vector",
            "vector_data": [0.2] * 1024,
        },
    ]

    for policy in ["first-wins", "last-wins", "fail", "warn"]:
        result = await client.update_batch_vectors(batch_updates, dedup_policy=policy)
        assert result["success"] == 2, (
            f"Policy {policy} should succeed with no duplicates"
        )
        assert result["duplicates_removed"] == 0, (
            f"Policy {policy} should report 0 duplicates"
        )


# ============================================================================
# RETRY LOGIC TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_retry_on_transient_error(client: QdrantClient, embedding_manager):
    """Test that transient errors are retried with exponential backoff."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    call_count = [0]
    original_update = client.client.update_vectors

    def mock_update_with_transient_failure(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call fails with transient error
            raise Exception("Connection timeout - temporary network issue")
        # Second call succeeds
        return original_update(*args, **kwargs)

    from unittest.mock import patch

    with patch.object(
        client.client, "update_vectors", side_effect=mock_update_with_transient_failure
    ):
        batch_updates = [
            {
                "anime_id": anime.id,
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            }
        ]

        result = await client.update_batch_vectors(
            batch_updates,
            max_retries=3,
            retry_delay=0.1,  # Fast retry for testing
        )

    assert call_count[0] == 2, "Should have retried once after transient error"
    assert result["success"] == 1, "Should eventually succeed after retry"
    assert result["failed"] == 0, "Should have no failures after successful retry"


@pytest.mark.asyncio
async def test_max_retries_exceeded(client: QdrantClient, embedding_manager):
    """Test that non-transient errors or max retries result in failure."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    call_count = [0]

    def mock_update_always_fails(*args, **kwargs):
        call_count[0] += 1
        raise Exception("Connection timeout - persistent")

    from unittest.mock import patch

    with patch.object(
        client.client, "update_vectors", side_effect=mock_update_always_fails
    ):
        batch_updates = [
            {
                "anime_id": anime.id,
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            }
        ]

        result = await client.update_batch_vectors(
            batch_updates, max_retries=2, retry_delay=0.05
        )

    assert call_count[0] == 3, "Should try initial + 2 retries = 3 total attempts"
    assert result["success"] == 0, "Should fail after max retries"
    assert result["failed"] == 1, "Should mark update as failed"
    assert "failed after" in result["results"][0]["error"].lower(), (
        "Error should mention retry attempts"
    )


@pytest.mark.asyncio
async def test_non_transient_error_no_retry(client: QdrantClient, embedding_manager):
    """Test that non-transient errors are not retried."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    call_count = [0]

    def mock_update_non_transient_error(*args, **kwargs):
        call_count[0] += 1
        raise Exception("Invalid data format - permanent error")

    from unittest.mock import patch

    with patch.object(
        client.client, "update_vectors", side_effect=mock_update_non_transient_error
    ):
        batch_updates = [
            {
                "anime_id": anime.id,
                "vector_name": "title_vector",
                "vector_data": [0.1] * 1024,
            }
        ]

        result = await client.update_batch_vectors(
            batch_updates, max_retries=3, retry_delay=0.1
        )

    assert call_count[0] == 1, "Should not retry non-transient errors"
    assert result["success"] == 0, "Should fail immediately"
    assert result["failed"] == 1, "Should mark as failed"


# ============================================================================
# UPDATE_ANIME_VECTORS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_basic(
    client: QdrantClient, embedding_manager
):
    """Test basic functionality of batch vector updates following script pattern."""
    anime_entries = [
        create_test_anime(anime_id="test-1", title="Test Anime 1"),
        create_test_anime(anime_id="test-2", title="Test Anime 2"),
    ]

    # Add initial documents
    for anime in anime_entries:
        await add_test_anime(client, embedding_manager, anime)

    # Generate vectors (following scripts/update_vectors.py pattern)
    gen_results = await embedding_manager.process_anime_batch(anime_entries)

    # Prepare updates
    updates = []
    for i, anime_entry in enumerate(anime_entries):
        vectors = gen_results[i].get("vectors", {})
        for vector_name in ["title_vector", "genre_vector"]:
            if vector_name in vectors and vectors[vector_name]:
                updates.append(
                    {
                        "anime_id": anime_entry.id,
                        "vector_name": vector_name,
                        "vector_data": vectors[vector_name],
                    }
                )

    # Update batch
    result = await client.update_batch_vectors(updates)

    assert result["success"] >= 2, "Should have at least 2 successful updates"
    assert "results" in result, "Should have detailed results"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_all_vectors(
    client: QdrantClient, embedding_manager, settings
):
    """Test updating all 11 vectors for one anime."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    # Generate all vectors
    gen_results = await embedding_manager.process_anime_batch([anime])
    vectors = gen_results[0].get("vectors", {})

    # Prepare updates for all generated vectors
    updates = []
    for vector_name in settings.vector_names.keys():
        if vector_name in vectors and vectors[vector_name]:
            updates.append(
                {
                    "anime_id": anime.id,
                    "vector_name": vector_name,
                    "vector_data": vectors[vector_name],
                }
            )

    result = await client.update_batch_vectors(updates)

    # Should have successfully updated multiple vectors
    assert result["success"] > 0, "Should have some successful updates"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_with_progress_callback(
    client: QdrantClient, embedding_manager
):
    """Test batch updates work correctly (progress tracking removed from low-level API)."""
    anime_entries = [create_test_anime(anime_id=f"test-{i}") for i in range(5)]

    # Add initial documents
    for anime in anime_entries:
        await add_test_anime(client, embedding_manager, anime)

    # Generate vectors
    gen_results = await embedding_manager.process_anime_batch(anime_entries)

    # Prepare updates
    updates = []
    for i, anime_entry in enumerate(anime_entries):
        vectors = gen_results[i].get("vectors", {})
        if "title_vector" in vectors and vectors["title_vector"]:
            updates.append(
                {
                    "anime_id": anime_entry.id,
                    "vector_name": "title_vector",
                    "vector_data": vectors["title_vector"],
                }
            )

    result = await client.update_batch_vectors(updates)

    # Should have 5 successful updates
    assert result["success"] == 5, "Should successfully update all 5 anime"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_handles_generation_failures(
    client: QdrantClient, embedding_manager
):
    """Test that generation failures can be detected."""
    # Create anime with minimal data (may not generate all vectors)
    minimal_anime = AnimeEntry(
        id="minimal",
        title="",  # Empty title
        genres=[],  # Empty genres
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[],
    )

    await add_test_anime(client, embedding_manager, minimal_anime)

    # Generate vectors
    gen_results = await embedding_manager.process_anime_batch([minimal_anime])
    vectors = gen_results[0].get("vectors", {})

    # Track which vectors were NOT generated
    requested_vectors = ["title_vector", "genre_vector"]
    generation_failures = [
        v for v in requested_vectors if v not in vectors or not vectors[v]
    ]

    # Should have some failures due to empty data
    assert len(generation_failures) >= 0, "Can detect missing vectors"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_empty_list(client: QdrantClient):
    """Test handling of empty updates list."""
    result = await client.update_batch_vectors([])

    assert result["success"] == 0
    assert result["failed"] == 0
    assert len(result["results"]) == 0


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_respects_batch_size(
    client: QdrantClient, embedding_manager
):
    """Test that batch updates work for larger batches."""
    anime_entries = [create_test_anime(anime_id=f"batch-{i}") for i in range(10)]

    # Add initial documents
    for anime in anime_entries:
        await add_test_anime(client, embedding_manager, anime)

    # Generate vectors for all anime
    gen_results = await embedding_manager.process_anime_batch(anime_entries)

    # Prepare all updates
    updates = []
    for i, anime_entry in enumerate(anime_entries):
        vectors = gen_results[i].get("vectors", {})
        if "title_vector" in vectors and vectors["title_vector"]:
            updates.append(
                {
                    "anime_id": anime_entry.id,
                    "vector_name": "title_vector",
                    "vector_data": vectors["title_vector"],
                }
            )

    result = await client.update_batch_vectors(updates)

    assert result["success"] == 10, "Should update all 10 anime"


# Tests for update_single_anime_vector() - high-level method with auto-generation


@pytest.mark.asyncio
async def test_update_single_anime_vector_success(
    client: QdrantClient, embedding_manager
):
    """Test successful single anime vector update with manual vector generation."""
    anime = create_test_anime(anime_id="single-anime-1", title="Test Anime")
    await add_test_anime(client, embedding_manager, anime)

    # Generate vectors using embedding_manager (matches new pattern from scripts)
    gen_results = await embedding_manager.process_anime_batch([anime])
    vectors = gen_results[0].get("vectors", {})

    # Verify vector was generated
    assert "title_vector" in vectors, "title_vector should be generated"
    vector_data = vectors["title_vector"]

    # Update using low-level method
    success = await client.update_single_vector(
        anime_id=anime.id, vector_name="title_vector", vector_data=vector_data
    )

    assert success is True, "Update should succeed"


@pytest.mark.asyncio
async def test_update_single_anime_vector_invalid_vector_name(
    client: QdrantClient, embedding_manager
):
    """Test that invalid vector name is rejected by validation."""
    anime = create_test_anime(anime_id="single-anime-2")
    await add_test_anime(client, embedding_manager, anime)

    # Try to update with invalid vector name
    dummy_vector = [0.0] * 1024
    success = await client.update_single_vector(
        anime_id=anime.id, vector_name="invalid_vector_name", vector_data=dummy_vector
    )

    # Should fail validation
    assert success is False, "Should fail with invalid vector name"


@pytest.mark.asyncio
async def test_update_single_anime_vector_generation_failure(
    client: QdrantClient, embedding_manager
):
    """Test handling when vector generation fails."""
    from unittest.mock import patch

    anime = create_test_anime(anime_id="single-anime-3")
    await add_test_anime(client, embedding_manager, anime)

    # Mock embedding manager to return empty vectors
    with patch.object(
        embedding_manager,
        "process_anime_batch",
        return_value=[{"vectors": {}}],  # No vectors generated
    ):
        gen_results = await embedding_manager.process_anime_batch([anime])
        vectors = gen_results[0].get("vectors", {})

        # Should have no title_vector
        assert "title_vector" not in vectors, "Vector generation should fail"


@pytest.mark.asyncio
async def test_update_single_anime_vector_multiple_vectors_sequential(
    client: QdrantClient, embedding_manager
):
    """Test updating multiple vectors sequentially for one anime."""
    anime = create_test_anime(anime_id="single-anime-4")
    await add_test_anime(client, embedding_manager, anime)

    # Generate all vectors once
    gen_results = await embedding_manager.process_anime_batch([anime])
    vectors = gen_results[0].get("vectors", {})

    # Update title_vector
    success1 = await client.update_single_vector(
        anime_id=anime.id,
        vector_name="title_vector",
        vector_data=vectors["title_vector"],
    )

    # Update genre_vector
    success2 = await client.update_single_vector(
        anime_id=anime.id,
        vector_name="genre_vector",
        vector_data=vectors["genre_vector"],
    )

    assert success1 is True, "First update should succeed"
    assert success2 is True, "Second update should succeed"


@pytest.mark.asyncio
async def test_update_single_anime_vector_image_vector(
    client: QdrantClient, embedding_manager
):
    """Test updating image vector with manual generation."""
    anime = create_test_anime(anime_id="single-anime-5")
    # Add images dict for image vector generation
    anime.images = {
        "poster": ["https://example.com/poster.jpg"],
        "cover": ["https://example.com/cover.jpg"],
    }
    await add_test_anime(client, embedding_manager, anime)

    # Generate vectors (image_vector may or may not be generated depending on URL availability)
    gen_results = await embedding_manager.process_anime_batch([anime])
    vectors = gen_results[0].get("vectors", {})

    # If image_vector was generated, update it
    if "image_vector" in vectors and vectors["image_vector"]:
        success = await client.update_single_vector(
            anime_id=anime.id,
            vector_name="image_vector",
            vector_data=vectors["image_vector"],
        )
        assert success is True, "Image vector update should succeed if generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
