#!/usr/bin/env python3
"""
Integration tests for QdrantClient.

INTEGRATION TESTS - These tests require:
- Real Qdrant database connection
- Real ML models (BGE-M3, OpenCLIP)
- Real data files
- Significant setup time (model loading)

Tests cover:
- update_single_point_vector and update_batch_point_vectors methods
- Vector generation with embedding_manager
- End-to-end workflows from anime data to Qdrant storage

Use pytest markers to run/skip integration tests:
    pytest -m "not integration"  # Skip integration tests
    pytest -m integration         # Run only integration tests
"""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.anime import Anime, AnimeRecord, AnimeStatus, AnimeType
from common.utils.id_generation import generate_deterministic_id
from qdrant_db import QdrantClient
from vector_db_interface import VectorDocument

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def test_anime_data():
    """Load test anime data from enriched database."""
    with open("./assets/seed_data/anime_database.json") as f:
        data = json.load(f)
    return data["data"][:3]  # Use first 3 anime for testing


def build_anime_record(
    *,
    anime_id: str,
    title: str,
    genres: list[str],
    year: int | None,
    type: AnimeType,
    status: AnimeStatus,
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


@pytest_asyncio.fixture
async def seeded_test_data(
    client: QdrantClient, embedding_manager, test_anime_data: list[dict]
):
    """Seed test anime data into Qdrant before running tests."""
    # Convert to AnimeRecord objects
    anime_entries = [AnimeRecord(**anime_dict) for anime_dict in test_anime_data]

    # Generate vectors using embedding_manager
    # Note: process_anime_batch returns a flattened list of all points (anime, characters, episodes)
    all_documents = await embedding_manager.process_anime_batch(anime_entries)

    # Upsert test data to ensure points exist
    result = await client.add_documents(all_documents, batch_size=len(all_documents))

    if not result.get("success"):
        pytest.skip("Failed to seed test data into Qdrant")

    # Return the anime entries for tests to use
    return anime_entries


def create_test_anime(
    point_id: str = "019bce3b-d48e-3d81-61ba-518ea655b2de", title: str = "Test Anime"
):
    """Create a test anime entry for isolated tests."""
    return build_anime_record(
        anime_id=point_id,
        title=title,
        genres=["Action", "Adventure"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )


async def add_test_anime(
    client: QdrantClient, embedding_manager, anime: AnimeRecord
) -> bool:
    """Add anime entry to Qdrant with generated vectors.

    Helper function that mimics the old add_documents([anime]) behavior.
    """
    gen_results = await embedding_manager.process_anime_vectors(anime)
    # The first document in results is always the anime point
    doc = gen_results[0]
    result = await client.add_documents([doc], batch_size=1)
    return result.get("success", False)


async def add_bare_anime_points(
    client: QdrantClient,
    anime_list: list[AnimeRecord] | AnimeRecord,
    batch_size: int = 100,
):
    """Add anime as bare points (empty vectors) for vector-update testing."""
    if isinstance(anime_list, AnimeRecord):
        anime_list = [anime_list]
    documents = []
    for anime_rec in anime_list:
        documents.append(
            VectorDocument(
                id=anime_rec.anime.id, vectors={}, payload=anime_rec.anime.model_dump()
            )
        )
    await client.add_documents(documents, batch_size=batch_size)


@pytest.mark.asyncio
async def test_update_single_vector(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test updating a single vector for one anime."""
    # Get first anime
    anime = seeded_test_data[0]

    # Generate text vector
    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)

    # Update single vector
    success = await client.update_single_point_vector(
        point_id=anime.anime.id, vector_name="text_vector", vector_data=text_vector
    )

    assert success is True, "Single vector update should succeed"


@pytest.mark.asyncio
async def test_update_single_vector_invalid_name(
    client: QdrantClient, seeded_test_data: list[AnimeRecord]
):
    """Test updating with invalid vector name."""
    anime = seeded_test_data[0]

    # Generate dummy vector
    dummy_vector = [0.0] * 1024

    # Try to update non-existent vector
    success = await client.update_single_point_vector(
        point_id=anime.anime.id, vector_name="invalid_vector", vector_data=dummy_vector
    )

    assert success is False, "Invalid vector name should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch updating multiple vectors."""
    batch_updates = []

    for anime in seeded_test_data:
        # Generate text vector
        full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
        text_vector = await embedding_manager.text_processor.encode_text(full_text)

        batch_updates.append(
            {
                "point_id": anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": text_vector,
            }
        )

    # Execute batch update
    result = await client.update_batch_point_vectors(batch_updates)

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
        assert "point_id" in update_result, "Result should contain point_id"
        assert "vector_name" in update_result, "Result should contain vector_name"


@pytest.mark.asyncio
async def test_update_batch_vectors_mixed(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch updating multiple vectors for same anime."""
    anime = seeded_test_data[0]

    # Generate multiple vectors
    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)

    batch_updates = [
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        }
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 1, "Vector update should succeed"
    assert result["failed"] == 0, "No updates should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_empty(client: QdrantClient):
    """Test batch update with empty list."""
    result = await client.update_batch_point_vectors([])

    assert result["success"] == 0, "Empty batch should have 0 successes"
    assert result["failed"] == 0, "Empty batch should have 0 failures"
    assert result["results"] == [], "Empty batch should have empty results list"


@pytest.mark.asyncio
async def test_update_image_vector(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test updating image vector."""
    anime = seeded_test_data[0]

    # Generate image vector
    image_urls = embedding_manager.field_mapper.extract_image_urls(anime.anime)
    if not image_urls:
        pytest.skip("No image data available for this anime")

    image_matrix = await embedding_manager.vision_processor.encode_images_batch(
        image_urls
    )

    if not image_matrix:
        pytest.skip("Failed to encode any images for this anime")

    # Update image vector (as multivector - needs to be list of vectors)
    success = await client.update_single_point_vector(
        point_id=anime.anime.id, vector_name="image_vector", vector_data=image_matrix
    )

    assert success is True, "Image vector update should succeed"


@pytest.mark.asyncio
async def test_update_all_text_vectors(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test updating text vectors for one anime."""
    anime = seeded_test_data[0]

    # Current architecture only has text_vector and image_vector
    text_vector_names = ["text_vector"]

    batch_updates = []

    for vector_name in text_vector_names:
        # Extract content
        content = embedding_manager.field_mapper.extract_anime_text(anime.anime)
        # Generate vector
        vector_data = await embedding_manager.text_processor.encode_text(content)

        batch_updates.append(
            {
                "point_id": anime.anime.id,
                "vector_name": vector_name,
                "vector_data": vector_data,
            }
        )

    # Execute batch update
    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == len(text_vector_names), (
        "Text vector updates should succeed"
    )
    assert result["failed"] == 0, "No updates should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_with_failures(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch update correctly tracks both successes and failures with detailed results."""
    anime = seeded_test_data[0]

    # Create batch with mix of valid and invalid updates
    batch_updates = []

    # Valid update
    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)
    batch_updates.append(
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        }
    )

    # Invalid update (invalid vector name)
    batch_updates.append(
        {
            "point_id": anime.anime.id,
            "vector_name": "invalid_vector_name",
            "vector_data": text_vector,
        }
    )

    result = await client.update_batch_point_vectors(batch_updates)

    # Verify result structure
    assert "success" in result, "Result should contain success count"
    assert "failed" in result, "Result should contain failed count"
    assert "results" in result, "Result should contain detailed results"

    # Verify counts
    assert result["success"] == 1, "Should have 1 successful update"
    assert result["failed"] == 1, "Should have 1 failed update"
    assert len(result["results"]) == 2, "Should have 2 detailed results"

    # Verify detailed results
    successful_updates = [r for r in result["results"] if r["success"]]
    failed_updates = [r for r in result["results"] if not r["success"]]

    assert len(successful_updates) == 1, "Should have 1 successful update detail"
    assert len(failed_updates) == 1, "Should have 1 failed update detail"

    # Check failed update has error message
    assert "error" in failed_updates[0], "Failed update should have error message"
    assert "invalid_vector_name" in str(failed_updates[0]["error"]).lower(), (
        "Error should mention invalid vector"
    )


@pytest.mark.asyncio
async def test_update_batch_vectors_all_validation_failures(
    client: QdrantClient, seeded_test_data: list[AnimeRecord]
):
    """Test batch update when all updates fail validation."""
    batch_updates = [
        # Invalid vector name
        {
            "point_id": seeded_test_data[0].anime.id,
            "vector_name": "invalid_vector_1",
            "vector_data": [0.1] * 1024,
        },
        # Wrong dimension
        {
            "point_id": seeded_test_data[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,  # Should be 1024
        },
        # Invalid data type
        {
            "point_id": seeded_test_data[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": "not a vector",
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 0, "All updates should fail"
    assert result["failed"] == 3, "Should have 3 failures"
    assert len(result["results"]) == 3, "Should have 3 detailed results"

    # All results should be failures with error messages
    for update_result in result["results"]:
        assert update_result["success"] is False, "All updates should fail"
        assert "error" in update_result, "Failed update should have error message"


@pytest.mark.asyncio
async def test_update_batch_vectors_partial_anime_success(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch update where some vectors succeed and some fail for the same anime."""
    anime = seeded_test_data[0]

    # Generate valid vectors
    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)

    # Since we use anime_id + vector_name as keys for deduplication, we can't easily test
    # partial success for the SAME vector name in one batch as one will be removed.
    # We'll use different vector names if available, or different anime.

    batch_updates = [
        # Anime 1 - valid
        {
            "point_id": seeded_test_data[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # Anime 1 - invalid (wrong dimension)
        {
            "point_id": seeded_test_data[0].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 512,  # Wrong dimension (should be 768)
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 1, "Should have 1 successful update"
    assert result["failed"] == 1, "Should have 1 failed update"
    assert len(result["results"]) == 2, "Should have 2 detailed results"


@pytest.mark.asyncio
async def test_update_batch_vectors_multiple_anime_mixed_results(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch update across multiple anime with mixed success/failure."""
    anime1, anime2, anime3 = seeded_test_data

    # Generate valid vectors
    text1 = embedding_manager.field_mapper.extract_anime_text(anime1.anime)
    text_vector1 = await embedding_manager.text_processor.encode_text(text1)

    text2 = embedding_manager.field_mapper.extract_anime_text(anime2.anime)
    text_vector2 = await embedding_manager.text_processor.encode_text(text2)

    batch_updates = [
        # Anime 1 - valid
        {
            "point_id": anime1.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector1,
        },
        # Anime 2 - valid
        {
            "point_id": anime2.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector2,
        },
        # Anime 3 - invalid (wrong dimension)
        {
            "point_id": anime3.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 2, "Should have 2 successful updates"
    assert result["failed"] == 1, "Should have 1 failed update"

    # Verify we can identify which anime had failures
    successful_anime_ids = {r["point_id"] for r in result["results"] if r["success"]}
    failed_anime_ids = {r["point_id"] for r in result["results"] if not r["success"]}

    assert anime1.anime.id in successful_anime_ids, "Anime 1 should succeed"
    assert anime2.anime.id in successful_anime_ids, "Anime 2 should succeed"
    assert anime3.anime.id in failed_anime_ids, "Anime 3 should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_dimension_validation(
    client: QdrantClient, seeded_test_data: list[AnimeRecord]
):
    """Test batch update validates vector dimensions correctly."""
    batch_updates = [
        # Too small
        {
            "point_id": seeded_test_data[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },
        # Too large (multivector format with wrong dimension)
        {
            "point_id": seeded_test_data[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [
                [0.1] * 2048
            ],  # Wrong dimension, but correct multivector format
        },
        # Empty vector (fails is_float_vector check)
        {
            "point_id": seeded_test_data[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": [],
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

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
    client: QdrantClient, seeded_test_data: list[AnimeRecord]
):
    """Test batch update rejects invalid data types."""
    anime = seeded_test_data[0]

    batch_updates = [
        # String instead of list
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": "not a vector",
        },
        # Dict instead of list
        {
            "point_id": anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": {"invalid": "data"},
        },
        # None value
        {
            "point_id": anime.anime.id,
            "vector_name": "invalid_vector_1",
            "vector_data": None,
        },
        # List with non-float values
        {
            "point_id": anime.anime.id,
            "vector_name": "invalid_vector_2",
            "vector_data": ["a", "b", "c"],
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 0, "All invalid data types should fail"
    assert result["failed"] == 4, "Should have 4 failures"

    # All should have error messages about invalid data
    for update_result in result["results"]:
        assert not update_result["success"], "Invalid data should fail"
        assert "error" in update_result, "Should have error message"


@pytest.mark.asyncio
async def test_update_batch_vectors_same_vector_multiple_updates(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch update with multiple updates to same vector (last one should win, deduplication occurs)."""
    anime = seeded_test_data[0]

    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector1 = await embedding_manager.text_processor.encode_text(full_text)
    text_vector2 = await embedding_manager.text_processor.encode_text(
        full_text
    )  # Same content, different instance

    batch_updates = [
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector1,
        },
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector2,
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Due to deduplication by anime_id + vector_name, only 1 update actually happens (last one wins)
    assert result["success"] == 1, "Should have 1 successful update (deduplicated)"
    assert result["failed"] == 0, "No failures expected"
    assert len(result["results"]) == 1, "Should have 1 result (deduplicated)"
    assert result["results"][0]["vector_name"] == "text_vector", "Should be text_vector"
    assert result["results"][0]["point_id"] == anime.anime.id, (
        "Should be correct anime_id"
    )


@pytest.mark.asyncio
async def test_update_batch_vectors_large_batch(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test batch update with many vectors per anime."""
    anime = seeded_test_data[0]

    # Current architecture has text_vector and image_vector
    vector_names = ["text_vector"]

    batch_updates = []
    for vector_name in vector_names:
        content = embedding_manager.field_mapper.extract_anime_text(anime.anime)
        vector_data = await embedding_manager.text_processor.encode_text(content)

        batch_updates.append(
            {
                "point_id": anime.anime.id,
                "vector_name": vector_name,
                "vector_data": vector_data,
            }
        )

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == len(vector_names), "Vector updates should succeed"
    assert result["failed"] == 0, "No failures expected"
    assert len(result["results"]) == len(vector_names), "Should have results for each"


@pytest.mark.asyncio
async def test_update_batch_vectors_preserves_order(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test that detailed results preserve order of input updates."""
    anime = seeded_test_data[0]

    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)

    # Mix of valid and invalid updates in specific order
    batch_updates = [
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },  # Valid
        {
            "point_id": anime.anime.id,
            "vector_name": "invalid_1",
            "vector_data": text_vector,
        },  # Invalid
        {
            "point_id": anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],  # Valid dim, multivector format
        },  # Valid
        {
            "point_id": anime.anime.id,
            "vector_name": "invalid_2",
            "vector_data": text_vector,
        },  # Invalid
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Results should maintain relationship to input (though not necessarily same order)
    assert len(result["results"]) == 4, "Should have 4 results"
    assert result["success"] == 2, "Should have 2 successes"
    assert result["failed"] == 2, "Should have 2 failures"

    # Count specific vectors in results
    result_vectors = [r["vector_name"] for r in result["results"]]
    assert "text_vector" in result_vectors, "text_vector should be in results"
    assert "image_vector" in result_vectors, "image_vector should be in results"
    assert "invalid_1" in result_vectors, "invalid_1 should be in results"
    assert "invalid_2" in result_vectors, "invalid_2 should be in results"


# ============================================================================
# DEDUPLICATION POLICY TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_deduplication_last_wins_policy(client: QdrantClient, embedding_manager):
    """Test that last-wins deduplication keeps the last occurrence."""
    anime_uuid = str(uuid.uuid4())
    anime = create_test_anime(point_id=anime_uuid)
    await add_test_anime(client, embedding_manager, anime)

    # Create duplicate updates with very different vectors
    first_vector = [0.001] * 1024
    second_vector = [0.999] * 1024  # Very different from first

    batch_updates = [
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": first_vector,
        },
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": second_vector,
        },  # Duplicate
    ]

    result = await client.update_batch_point_vectors(
        batch_updates, dedup_policy="last-wins"
    )

    assert result["success"] == 1, "Should have 1 successful update (last one)"
    assert result["duplicates_removed"] == 1, "Should have removed 1 duplicate"

    # Verify the second vector was used by searching with it
    search_result = await client.search_single_vector(
        vector_name="text_vector", vector_data=second_vector, limit=5
    )

    # Find our anime in results
    # Search result 'id' is overwritten by payload anime ID
    our_anime = next((r for r in search_result if r["id"] == anime.anime.id), None)
    assert our_anime is not None, f"Should find anime {anime.anime.id} in results"
    assert our_anime["similarity_score"] > 0.99, (
        "Should match the last vector with high similarity"
    )


@pytest.mark.asyncio
async def test_deduplication_first_wins_policy(client: QdrantClient, embedding_manager):
    """Test that first-wins deduplication keeps the first occurrence."""
    anime_uuid = str(uuid.uuid4())
    anime = create_test_anime(point_id=anime_uuid)
    await add_test_anime(client, embedding_manager, anime)

    first_vector = [0.001] * 1024
    second_vector = [0.999] * 1024

    batch_updates = [
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": first_vector,
        },
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": second_vector,
        },  # Duplicate
    ]

    result = await client.update_batch_point_vectors(
        batch_updates, dedup_policy="first-wins"
    )

    assert result["success"] == 1, "Should have 1 successful update (first one)"
    assert result["duplicates_removed"] == 1, "Should have removed 1 duplicate"

    # Verify the first vector was used
    search_result = await client.search_single_vector(
        vector_name="text_vector", vector_data=first_vector, limit=5
    )
    # Find our anime in results
    # Search result 'id' is overwritten by payload anime ID
    our_anime = next((r for r in search_result if r["id"] == anime.anime.id), None)
    assert our_anime is not None, f"Should find anime {anime.anime.id} in results"
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
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.9] * 1024,
        },  # Duplicate
    ]

    with pytest.raises(ValueError, match="Duplicate update found"):
        await client.update_batch_point_vectors(batch_updates, dedup_policy="fail")


@pytest.mark.asyncio
async def test_deduplication_warn_policy(
    client: QdrantClient, embedding_manager, caplog
):
    """Test that warn policy logs warning but continues with last-wins."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    batch_updates = [
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.9] * 1024,
        },  # Duplicate
    ]

    import logging

    with caplog.at_level(logging.WARNING):
        result = await client.update_batch_point_vectors(
            batch_updates, dedup_policy="warn"
        )

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
            "point_id": anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "point_id": anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.2] * 768],  # Multivector format
        },
    ]

    for policy in ["first-wins", "last-wins", "fail", "warn"]:
        result = await client.update_batch_point_vectors(
            batch_updates, dedup_policy=policy
        )
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

    def mock_update_with_transient_failure(*_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call fails with transient error
            raise ConnectionError("Connection timeout")
        # Second call succeeds
        return original_update(*_args, **_kwargs)

    from unittest.mock import patch

    with patch.object(
        client.client, "update_vectors", side_effect=mock_update_with_transient_failure
    ):
        batch_updates = [
            {
                "point_id": anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            }
        ]

        result = await client.update_batch_point_vectors(
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

    def mock_always_fails(*_args, **_kwargs):
        call_count[0] += 1
        raise RuntimeError("Connection timeout - persistent")

    from unittest.mock import patch

    # Patch qdrant_client.AsyncQdrantClient.update_vectors which is what client.client is
    with patch.object(client.client, "update_vectors", side_effect=mock_always_fails):
        batch_updates = [
            {
                "point_id": anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            }
        ]

        result = await client.update_batch_point_vectors(
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

    def mock_non_transient_error(*_args, **_kwargs):
        call_count[0] += 1
        raise RuntimeError("Invalid data format - permanent error")

    from unittest.mock import patch

    with patch.object(
        client.client, "update_vectors", side_effect=mock_non_transient_error
    ):
        batch_updates = [
            {
                "point_id": anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": [0.1] * 1024,
            }
        ]

        result = await client.update_batch_point_vectors(
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
    # Create test data
    anime_entries = [
        create_test_anime(point_id=str(uuid.uuid4()), title="Test Anime 1"),
        create_test_anime(point_id=str(uuid.uuid4()), title="Test Anime 2"),
    ]

    # Add initial documents
    for anime in anime_entries:
        await add_test_anime(client, embedding_manager, anime)

    # Generate vectors
    # Parent anime points are the first in results for each record
    updates = []
    for anime_entry in anime_entries:
        gen_results = await embedding_manager.process_anime_vectors(anime_entry)
        anime_doc = gen_results[0]
        vectors = anime_doc.vectors
        if "text_vector" in vectors:
            updates.append(
                {
                    "point_id": anime_entry.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vectors["text_vector"],
                }
            )

    # Update batch
    result = await client.update_batch_point_vectors(updates)

    assert result["success"] >= 2, "Should have at least 2 successful updates"
    assert "results" in result, "Should have detailed results"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_all_vectors(
    client: QdrantClient, embedding_manager, settings
):
    """Test updating all vectors for one anime."""
    anime = create_test_anime()
    await add_test_anime(client, embedding_manager, anime)

    # Generate all vectors
    gen_results = await embedding_manager.process_anime_vectors(anime)
    anime_doc = gen_results[0]
    vectors = anime_doc.vectors

    # Prepare updates for all generated vectors
    updates = []
    for vector_name in settings.vector_names.keys():
        if vectors.get(vector_name):
            # If it's a multivector (like image_vector), we take the first one for this test
            # as update_single_vector expects a single vector
            vec_data = vectors[vector_name]
            if isinstance(vec_data[0], list):  # list of lists
                vec_data = vec_data[0]

            updates.append(
                {
                    "point_id": anime.anime.id,
                    "vector_name": vector_name,
                    "vector_data": vec_data,
                }
            )

    result = await client.update_batch_point_vectors(updates)

    # Should have successfully updated multiple vectors
    assert result["success"] > 0, "Should have some successful updates"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_with_progress_callback(
    client: QdrantClient, embedding_manager
):
    """Test batch updates work correctly."""
    # Create 2 test anime
    anime_entries = [create_test_anime(point_id=str(uuid.uuid4())) for _ in range(2)]

    # Add initial documents
    for anime in anime_entries:
        await add_test_anime(client, embedding_manager, anime)

    # Generate vectors
    updates = []
    for anime_entry in anime_entries:
        gen_results = await embedding_manager.process_anime_vectors(anime_entry)
        vectors = gen_results[0].vectors
        if "text_vector" in vectors:
            updates.append(
                {
                    "point_id": anime_entry.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vectors["text_vector"],
                }
            )

    result = await client.update_batch_point_vectors(updates)

    # Should have 2 successful updates
    assert result["success"] == 2, "Should successfully update all 2 anime"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_handles_generation_failures(
    client: QdrantClient, embedding_manager
):
    """Test that generation failures can be handled."""
    # Create anime with minimal data
    minimal_anime = build_anime_record(
        anime_id="minimal",
        title="",  # Empty title
        genres=[],  # Empty genres
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_test_anime(client, embedding_manager, minimal_anime)

    # Generate vectors
    gen_results = await embedding_manager.process_anime_vectors(minimal_anime)
    vectors = gen_results[0].vectors

    # Track which vectors were NOT generated
    # text_vector is always generated (at least as zero vector)
    requested_vectors = ["text_vector"]
    generation_failures = [
        v for v in requested_vectors if v not in vectors or not vectors[v]
    ]

    # Should have no failures for text_vector as it falls back to zero
    assert len(generation_failures) == 0, "Should handle minimal data gracefully"


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_empty_list(client: QdrantClient):
    """Test handling of empty updates list."""
    result = await client.update_batch_point_vectors([])

    assert result["success"] == 0
    assert result["failed"] == 0
    assert len(result["results"]) == 0


@pytest.mark.asyncio
async def test_update_batch_anime_vectors_respects_batch_size(
    client: QdrantClient, embedding_manager
):
    """Test that batch updates work for larger batches."""
    anime_entries = [create_test_anime(point_id=str(uuid.uuid4())) for i in range(10)]

    # Add initial documents
    for anime in anime_entries:
        await add_test_anime(client, embedding_manager, anime)

    # Generate vectors for all anime
    updates = []
    for anime_entry in anime_entries:
        gen_results = await embedding_manager.process_anime_vectors(anime_entry)
        vectors = gen_results[0].vectors
        if "text_vector" in vectors:
            updates.append(
                {
                    "point_id": anime_entry.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vectors["text_vector"],
                }
            )

    result = await client.update_batch_point_vectors(updates)

    assert result["success"] == 10, "Should update all 10 anime"


# Tests for update_single_anime_vector() - high-level method with auto-generation


@pytest.mark.asyncio
async def test_update_single_anime_vector_success(
    client: QdrantClient, embedding_manager
):
    """Test successful single anime vector update with manual vector generation."""
    anime = create_test_anime(point_id=str(uuid.uuid4()), title="Test Anime")
    await add_test_anime(client, embedding_manager, anime)

    # Generate vectors using embedding_manager
    gen_results = await embedding_manager.process_anime_vectors(anime)
    vectors = gen_results[0].vectors

    # Verify vector was generated
    assert "text_vector" in vectors, "text_vector should be generated"
    vector_data = vectors["text_vector"]

    # Update using low-level method
    success = await client.update_single_point_vector(
        point_id=anime.anime.id, vector_name="text_vector", vector_data=vector_data
    )

    assert success is True, "Update should succeed"


@pytest.mark.asyncio
async def test_update_single_anime_vector_invalid_vector_name(
    client: QdrantClient, embedding_manager
):
    """Test that invalid vector name is rejected by validation."""
    anime = create_test_anime(point_id=str(uuid.uuid4()))
    await add_test_anime(client, embedding_manager, anime)

    # Try to update with invalid vector name
    dummy_vector = [0.0] * 1024
    success = await client.update_single_point_vector(
        point_id=anime.anime.id,
        vector_name="invalid_vector_name",
        vector_data=dummy_vector,
    )

    # Should fail validation
    assert success is False, "Should fail with invalid vector name"


@pytest.mark.asyncio
async def test_update_single_anime_vector_generation_failure(
    client: QdrantClient, embedding_manager
):
    """Test handling of vector generation failures."""
    anime = create_test_anime(point_id=str(uuid.uuid4()))
    await add_test_anime(client, embedding_manager, anime)

    # Mock embedding manager to return empty vectors
    with patch.object(
        embedding_manager,
        "process_anime_vectors",
        return_value=[
            VectorDocument(id="test", vectors={}, payload={})
        ],  # No vectors generated
    ):
        gen_results = await embedding_manager.process_anime_vectors(anime)
        vectors = gen_results[0].vectors

        # Should have no text_vector
        assert "text_vector" not in vectors, "Vector generation should fail"


@pytest.mark.asyncio
async def test_update_single_anime_vector_multiple_vectors_sequential(
    client: QdrantClient, embedding_manager
):
    """Test updating multiple vectors sequentially for one anime."""
    anime = create_test_anime(point_id=str(uuid.uuid4()))
    await add_test_anime(client, embedding_manager, anime)

    # Generate all vectors once
    gen_results = await embedding_manager.process_anime_vectors(anime)
    vectors = gen_results[0].vectors

    # Update text_vector
    success1 = await client.update_single_point_vector(
        point_id=anime.anime.id,
        vector_name="text_vector",
        vector_data=vectors["text_vector"],
    )

    assert success1 is True, "First update should succeed"


@pytest.mark.asyncio
async def test_update_single_anime_vector_image_vector(
    client: QdrantClient, embedding_manager
):
    """Test updating image vector with manual generation."""
    anime = create_test_anime(point_id=str(uuid.uuid4()))
    # Add images dict for image vector generation
    anime.anime.images = {
        "covers": ["https://example.com/poster.jpg"],
        "posters": ["https://example.com/cover.jpg"],
    }
    await add_test_anime(client, embedding_manager, anime)

    # Generate vectors
    gen_results = await embedding_manager.process_anime_vectors(anime)
    vectors = gen_results[0].vectors

    # If image_vector was generated, update it
    if vectors.get("image_vector"):
        # Update single vector expects a single vector, not a matrix
        # For image_vector (multivector), we update with one of the vectors
        image_data = vectors["image_vector"]
        if isinstance(image_data[0], list):
            image_data = image_data[0]

        success = await client.update_single_point_vector(
            point_id=anime.anime.id,
            vector_name="image_vector",
            vector_data=image_data,
        )
        assert success is True, "Image vector update should succeed if generated"


# ============================================================================
# NEW TESTS FOR GENERIC update_single_point_vector METHOD
# ============================================================================


@pytest.mark.asyncio
async def test_update_single_point_vector_for_character(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test updating a vector for a character point (not anime point)."""
    # Get anime with characters
    anime_with_chars = None
    for anime in seeded_test_data:
        if anime.characters and len(anime.characters) > 0:
            anime_with_chars = anime
            break

    if not anime_with_chars:
        pytest.skip("No anime with characters in test data")

    character = anime_with_chars.characters[0]
    char_point_id = character.id or generate_deterministic_id(
        f"{anime_with_chars.anime.id}_{character.name}"
    )

    # Generate text vector for character
    char_text = f"{character.name} {character.description or ''}"
    text_vector = await embedding_manager.text_processor.encode_text(char_text)

    # Update character point's text vector using generic method
    success = await client.update_single_point_vector(
        point_id=char_point_id,  # Character ID, not anime ID
        vector_name="text_vector",
        vector_data=text_vector,
    )

    assert success is True, "Character point vector update should succeed"


@pytest.mark.asyncio
async def test_update_single_point_vector_for_episode(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test updating a vector for an episode point (not anime point)."""
    # Get anime with episodes
    anime_with_eps = None
    for anime in seeded_test_data:
        if anime.episodes and len(anime.episodes) > 0:
            anime_with_eps = anime
            break

    if not anime_with_eps:
        pytest.skip("No anime with episodes in test data")

    episode = anime_with_eps.episodes[0]
    ep_point_id = episode.id or generate_deterministic_id(
        f"{anime_with_eps.anime.id}_{episode.episode_number}"
    )

    # Generate text vector for episode
    ep_text = f"Episode {episode.episode_number}: {episode.title or ''} {episode.description or ''}"
    text_vector = await embedding_manager.text_processor.encode_text(ep_text)

    # Update episode point's text vector using generic method
    success = await client.update_single_point_vector(
        point_id=ep_point_id,  # Episode ID, not anime ID
        vector_name="text_vector",
        vector_data=text_vector,
    )

    assert success is True, "Episode point vector update should succeed"


@pytest.mark.asyncio
async def test_update_single_point_vector_for_anime(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test that update_single_point_vector still works for anime points."""
    anime = seeded_test_data[0]

    # Generate text vector
    full_text = embedding_manager.field_mapper.extract_anime_text(anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)

    # Update anime point using generic method
    success = await client.update_single_point_vector(
        point_id=anime.anime.id,  # Anime ID
        vector_name="text_vector",
        vector_data=text_vector,
    )

    assert success is True, "Anime point vector update should succeed"


@pytest.mark.asyncio
async def test_update_single_point_vector_with_multivector(
    client: QdrantClient, embedding_manager, seeded_test_data: list[AnimeRecord]
):
    """Test updating a multivector (image_vector) for a character point."""
    # Get anime with characters that have images
    anime_with_char_images = None
    for anime in seeded_test_data:
        if anime.characters:
            for char in anime.characters:
                if char.images and len(char.images) > 0:
                    anime_with_char_images = anime
                    break
        if anime_with_char_images:
            break

    if not anime_with_char_images:
        pytest.skip("No characters with images in test data")

    character = next(
        c for c in anime_with_char_images.characters if c.images and len(c.images) > 0
    )
    char_point_id = character.id or generate_deterministic_id(
        f"{anime_with_char_images.anime.id}_{character.name}"
    )

    # Generate image vectors for character
    image_urls = [character.images[0]]
    image_vectors = await embedding_manager.vision_processor.encode_images_batch(
        image_urls
    )

    if not image_vectors:
        pytest.skip("Failed to generate image vectors")

    # Update character point's image vector (multivector)
    success = await client.update_single_point_vector(
        point_id=char_point_id,
        vector_name="image_vector",
        vector_data=image_vectors,  # List of vectors (multivector)
    )

    assert success is True, "Character image multivector update should succeed"


# ============================================================================
# BATCH UPDATE SCENARIOS & EDGE CASES
# ============================================================================


@pytest.mark.asyncio
async def test_vector_persistence_after_update(client: QdrantClient, embedding_manager):
    """Test that vectors are actually persisted and retrievable after update."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Persistence Test Anime",
        genres=["Action"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [test_anime], batch_size=1)

    # Generate and update text_vector
    title_content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(title_content)
    assert text_vector is not None, "encode_text should return valid vector"

    batch_updates = [
        {
            "point_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        }
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Verify update succeeded
    assert result["success"] == 1, "Update should succeed"
    assert result["failed"] == 0, "No failures expected"

    # Verify vector is retrievable via search
    query_vector = await embedding_manager.text_processor.encode_text(
        "Persistence Test Anime"
    )
    assert query_vector is not None, "encode_text should return valid vector"

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
    client: QdrantClient, embedding_manager
):
    """Test that detailed results from update_batch_point_vectors enable accurate per-update tracking."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(3)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    # Create batch with mix of valid and invalid updates
    batch_updates = []

    # Valid update for anime-0
    title_content = embedding_manager.field_mapper.extract_anime_text(
        test_anime[0].anime
    )
    text_vector = await embedding_manager.text_processor.encode_text(title_content)
    assert text_vector is not None
    batch_updates.append(
        {
            "point_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        }
    )

    # Invalid update (wrong dimension) for anime-1
    batch_updates.append(
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,  # Wrong dimension
        }
    )

    # Valid update for anime-2
    title_content2 = embedding_manager.field_mapper.extract_anime_text(
        test_anime[2].anime
    )
    text_vector2 = await embedding_manager.text_processor.encode_text(title_content2)
    assert text_vector2 is not None
    batch_updates.append(
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector2,
        }
    )

    result = await client.update_batch_point_vectors(batch_updates)

    # Verify detailed results enable accurate tracking
    assert "results" in result, "Should have detailed results"
    assert len(result["results"]) == 3, "Should have 3 detailed results"

    # Verify we can identify which specific anime failed
    failed_anime_ids = [r["point_id"] for r in result["results"] if not r["success"]]
    successful_anime_ids = [r["point_id"] for r in result["results"] if r["success"]]

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
    client: QdrantClient, embedding_manager
):
    """Test that an anime is only counted as successful if ALL its requested vectors were updated."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="All or Nothing Test",
        genres=["Action"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [test_anime], batch_size=1)

    # We'll request 3 vectors, but only 2 will be valid names
    full_text = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(full_text)
    assert text_vector is not None
    image_vector = [[0.1] * 768]

    batch_updates = [
        # 1. Valid text_vector
        {
            "point_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # 2. Valid image_vector (multivector format)
        {
            "point_id": test_anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": image_vector,
        },
        # 3. Invalid vector name
        {
            "point_id": test_anime.anime.id,
            "vector_name": "invalid_vector",
            "vector_data": text_vector,
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Success tracking should show 2 vector successes but 0 anime successes
    # because one vector for the anime failed.
    assert result["success"] == 2, "2 individual vectors should have succeeded"
    assert result["failed"] == 1, "1 individual vector should have failed"

    # The aggregation logic (which we're testing via scripts/update_vectors.py's expected usage)
    # would see this anime as failed.
    failed_anime_ids = {r["point_id"] for r in result["results"] if not r["success"]}
    assert test_anime.anime.id in failed_anime_ids, "Anime should be in failed set"


@pytest.mark.asyncio
async def test_per_vector_statistics_from_detailed_results(
    client: QdrantClient, embedding_manager
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
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(3)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    # Create updates with specific failure pattern:
    # - text_vector: all 3 succeed
    # - image_vector: 2 succeed, 1 fails (anime-1)
    batch_updates = []

    for i, anime_rec in enumerate(test_anime):
        # Text vector - all valid
        title_content = embedding_manager.field_mapper.extract_anime_text(
            anime_rec.anime
        )
        text_vector = await embedding_manager.text_processor.encode_text(title_content)
        assert text_vector is not None
        batch_updates.append(
            {
                "point_id": anime_rec.anime.id,
                "vector_name": "text_vector",
                "vector_data": text_vector,
            }
        )

        # Image vector - anime-1 has invalid dimension
        if i == 1:
            batch_updates.append(
                {
                    "point_id": anime_rec.anime.id,
                    "vector_name": "image_vector",
                    "vector_data": [0.1] * 512,  # Invalid
                }
            )
        else:
            image_vector = [[0.1] * 768]  # Multivector format
            batch_updates.append(
                {
                    "point_id": anime_rec.anime.id,
                    "vector_name": "image_vector",
                    "vector_data": image_vector,
                }
            )

    result = await client.update_batch_point_vectors(batch_updates)

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
async def test_mixed_batch_all_combinations(client: QdrantClient, embedding_manager):
    """Test batch with all possible anime success combinations: all pass, partial pass, all fail."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Mixed Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(5)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    batch_updates = []

    # Anime 0: 2/2 vectors succeed (all pass)
    for vector_name in ["text_vector", "image_vector"]:
        if vector_name == "text_vector":
            content = embedding_manager.field_mapper.extract_anime_text(
                test_anime[0].anime
            )
            vector = await embedding_manager.text_processor.encode_text(content)
            assert vector is not None
        else:
            vector = [[0.1] * 768]  # Multivector format
        batch_updates.append(
            {
                "point_id": test_anime[0].anime.id,
                "vector_name": vector_name,
                "vector_data": vector,
            }
        )

    # Anime 1: 1/2 vectors succeed (partial pass)
    batch_updates.append(
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        }
    )
    batch_updates.append(
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 512,
        }
    )  # Fail (wrong dim)

    # Anime 2: 0/2 vectors succeed (all fail)
    batch_updates.append(
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        }
    )  # Fail
    batch_updates.append(
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "image_vector",
            "vector_data": "invalid",
        }
    )  # Fail

    # Anime 3: 0/2 vectors succeed (invalid name + invalid data)
    batch_updates.append(
        {
            "point_id": test_anime[3].anime.id,
            "vector_name": "invalid_vector",
            "vector_data": [0.1] * 1024,
        }
    )
    batch_updates.append(
        {
            "point_id": test_anime[3].anime.id,
            "vector_name": "image_vector",
            "vector_data": None,
        }
    )

    # Anime 4: 2/2 vectors succeed (all pass)
    for vector_name in ["text_vector", "image_vector"]:
        if vector_name == "text_vector":
            content = embedding_manager.field_mapper.extract_anime_text(
                test_anime[4].anime
            )
            vector = await embedding_manager.text_processor.encode_text(content)
            assert vector is not None
        else:
            vector = [[0.1] * 768]  # Multivector format
        batch_updates.append(
            {
                "point_id": test_anime[4].anime.id,
                "vector_name": vector_name,
                "vector_data": vector,
            }
        )

    result = await client.update_batch_point_vectors(batch_updates)

    # Calculate anime success map
    anime_success_map = {}
    for r in result["results"]:
        anime_id = r["point_id"]
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
    client: QdrantClient, embedding_manager
):
    """Test updating all supported vector types in a single batch."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="All Vectors Test",
        genres=["Action", "Drama"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [test_anime], batch_size=1)

    # All supported vector names
    vector_names = [
        "text_vector",
        "image_vector",
    ]

    batch_updates = []
    for vector_name in vector_names:
        if "image" in vector_name:
            # Image vectors are 768-dimensional (multivector format)
            vector_data: list[float] | list[list[float]] = [[0.1] * 768]
        else:
            # Text vectors are 1024-dimensional
            content = embedding_manager.field_mapper.extract_anime_text(
                test_anime.anime
            )
            vector_data_opt = await embedding_manager.text_processor.encode_text(
                content
            )
            assert vector_data_opt is not None
            vector_data = vector_data_opt

        batch_updates.append(
            {
                "point_id": test_anime.anime.id,
                "vector_name": vector_name,
                "vector_data": vector_data,
            }
        )

    result = await client.update_batch_point_vectors(batch_updates)

    # Verify all supported vectors were processed
    assert len(result["results"]) == 2, "Should have 2 detailed results"
    assert result["success"] == 2, "All updates should succeed"
    assert result["failed"] == 0, "No failures expected"

    # Verify each vector name is present in results
    processed_names = {r["vector_name"] for r in result["results"]}
    for name in vector_names:
        assert name in processed_names, f"Vector {name} should be in results"


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
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(5)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    batch_updates = [
        # Exactly 1024 (should pass)
        {
            "point_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        # Exactly 768 (should pass for image vectors, multivector format)
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],
        },
        # Wrong dimension for text_vector
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },
        # Wrong dimension for image_vector
        {
            "point_id": test_anime[3].anime.id,
            "vector_name": "image_vector",
            "vector_data": [0.1] * 1024,
        },
        # Empty vector
        {
            "point_id": test_anime[4].anime.id,
            "vector_name": "text_vector",
            "vector_data": [],
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 2, "Exactly correct sizes should succeed"
    assert result["failed"] == 3, "3 dimension/validation mismatches should fail"


@pytest.mark.asyncio
async def test_non_existent_anime_ids(client: QdrantClient):
    """Test updates to anime IDs that don't exist in Qdrant."""
    batch_updates = [
        {
            "point_id": "non-existent-1",
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "point_id": "non-existent-2",
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
        {
            "point_id": "non-existent-3",
            "vector_name": "text_vector",
            "vector_data": [0.1] * 1024,
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Qdrant's update_vectors accepts updates for non-existent points without error,
    # so our client reports all 3 as successful (no exception raised by Qdrant).
    assert result["success"] == 3, "All updates should succeed (Qdrant doesn't reject non-existent IDs)"
    assert result["failed"] == 0, "No updates should fail"
    assert len(result["results"]) == 3, "Should have one result per update"
    assert all(r["success"] for r in result["results"]), "Each result should be marked successful"


@pytest.mark.asyncio
async def test_mixed_valid_invalid_anime_ids(client: QdrantClient, embedding_manager):
    """Test batch with mix of existing and non-existing anime IDs.

    Note: Qdrant's update_vectors operation updates existing points.
    Non-existing points will be processed but may not be updated if they don't exist.
    """
    existing_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Existing Anime",
        genres=["Action"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [existing_anime], batch_size=1)

    content = embedding_manager.field_mapper.extract_anime_text(existing_anime.anime)
    vector = await embedding_manager.text_processor.encode_text(content)
    assert vector is not None

    batch_updates = [
        # Existing anime - should work
        {
            "point_id": existing_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        },
        # Another existing update with different vector name to avoid deduplication
        {
            "point_id": existing_anime.anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],  # Multivector format
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Both existing anime updates should work
    assert result["success"] == 2, "Existing anime updates should work"
    assert result["failed"] == 0, "No failures for existing anime"


@pytest.mark.asyncio
async def test_results_ordering_matches_input(client: QdrantClient, embedding_manager):
    """Test that detailed results maintain relationship to input order."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Order Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(5)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    # Create specific pattern: success, fail, success, fail, success
    batch_updates = []
    for i, anime_rec in enumerate(test_anime):
        vector: list[float]
        if i % 2 == 0:  # Even indices - valid
            content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
            vector_opt = await embedding_manager.text_processor.encode_text(content)
            assert vector_opt is not None
            vector = vector_opt
        else:  # Odd indices - invalid
            vector = [0.1] * 512  # Wrong dimension

        batch_updates.append(
            {
                "point_id": anime_rec.anime.id,
                "vector_name": "text_vector",
                "vector_data": vector,
            }
        )

    result = await client.update_batch_point_vectors(batch_updates)

    # Verify we can match results to input
    assert len(result["results"]) == 5, "Should have 5 results"

    # Check that all anime IDs from input are in results
    input_anime_ids = {u["point_id"] for u in batch_updates}
    result_anime_ids = {r["point_id"] for r in result["results"]}

    assert input_anime_ids == result_anime_ids, "All anime IDs should be in results"


@pytest.mark.asyncio
async def test_large_batch_realistic_failures(client: QdrantClient, embedding_manager):
    """Test large batch (100 anime) with realistic failure patterns."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Large Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(100)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=50)

    batch_updates = []
    for i, anime_rec in enumerate(test_anime):
        content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
        vector = await embedding_manager.text_processor.encode_text(content)
        assert vector is not None

        # 90% success rate - realistic scenario
        if i % 10 != 0:  # 90% valid
            batch_updates.append(
                {
                    "point_id": anime_rec.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vector,
                }
            )
        else:  # 10% invalid
            batch_updates.append(
                {
                    "point_id": anime_rec.anime.id,
                    "vector_name": "text_vector",
                    "vector_data": [0.1] * 512,
                }
            )

    result = await client.update_batch_point_vectors(batch_updates)

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
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [test_anime], batch_size=1)

    # Update same vector 3 times sequentially
    for i in range(3):
        vector = [float(i)] * 1024  # Different values each time
        batch_updates = [
            {
                "point_id": test_anime.anime.id,
                "vector_name": "text_vector",
                "vector_data": vector,
            }
        ]

        result = await client.update_batch_point_vectors(batch_updates)
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
async def test_image_and_text_vectors_mixed(client: QdrantClient, embedding_manager):
    """Test mixing image vectors (768-dim) and text vectors (1024-dim) in same batch."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Mixed Dim Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(4)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    content = embedding_manager.field_mapper.extract_anime_text(test_anime[0].anime)
    text_vector = await embedding_manager.text_processor.encode_text(content)
    assert text_vector is not None

    batch_updates = [
        # Text vector (1024-dim)
        {
            "point_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # Image vector (768-dim, multivector format)
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],
        },
        # Another text vector
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": text_vector,
        },
        # Another image vector (768-dim, multivector format)
        {
            "point_id": test_anime[3].anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["success"] == 4, "All mixed dimension updates should succeed"
    assert result["failed"] == 0, "No failures expected"


@pytest.mark.asyncio
async def test_batch_size_boundaries(client: QdrantClient, embedding_manager):
    """Test various batch sizes: 1, 2, 50, 100, 500."""
    # Create test anime
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(500)
    ]

    # Add in batches to avoid memory issues
    for i in range(0, 500, 100):
        await add_bare_anime_points(client, test_anime[i : i + 100], batch_size=100)

    # Test different batch sizes
    batch_sizes_to_test = [1, 2, 50, 100, 500]

    for batch_size in batch_sizes_to_test:
        batch_updates = []
        for i in range(batch_size):
            content = embedding_manager.field_mapper.extract_anime_text(
                test_anime[i].anime
            )
            vector = await embedding_manager.text_processor.encode_text(content)
            assert vector is not None
            batch_updates.append(
                {
                    "point_id": test_anime[i].anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vector,
                }
            )

        result = await client.update_batch_point_vectors(batch_updates)

        assert result["success"] == batch_size, (
            f"Batch size {batch_size} should have {batch_size} successes"
        )
        assert result["failed"] == 0, f"Batch size {batch_size} should have 0 failures"
        assert len(result["results"]) == batch_size, f"Should have {batch_size} results"


@pytest.mark.asyncio
async def test_update_then_search_consistency(client: QdrantClient, embedding_manager):
    """Test that updated vectors are immediately searchable with correct results."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Consistency Test Anime",
        genres=["Action", "Adventure"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [test_anime], batch_size=1)

    # Update text_vector
    content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    text_vector = await embedding_manager.text_processor.encode_text(content)
    assert text_vector is not None

    update_result = await client.update_batch_point_vectors(
        [
            {
                "point_id": test_anime.anime.id,
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
    assert test_anime.anime.id in found_ids, (
        "Updated anime should be immediately searchable"
    )

    # Should be top result or very high in results (might have other test data)
    # Self-similarity should be very high
    our_result = next(
        (r for r in search_results if r["id"] == test_anime.anime.id), None
    )
    assert our_result is not None, "Must find updated anime"
    assert our_result["similarity_score"] > 0.98, "Self-similarity should be very high"


@pytest.mark.asyncio
async def test_similarity_search_after_multiple_updates(
    client: QdrantClient, embedding_manager
):
    """Test that similarity search returns correct results after vector updates."""
    # Create 3 similar anime
    anime_list = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title="Action Hero Adventure",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        ),
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title="Action Hero Story",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        ),
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title="Romance Comedy Love",
            genres=["Romance"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        ),
    ]

    await add_bare_anime_points(client, anime_list, batch_size=3)

    # Update all title vectors
    batch_updates = []
    for anime_rec in anime_list:
        content = embedding_manager.field_mapper.extract_anime_text(anime_rec.anime)
        vector = await embedding_manager.text_processor.encode_text(content)
        assert vector is not None
        batch_updates.append(
            {
                "point_id": anime_rec.anime.id,
                "vector_name": "text_vector",
                "vector_data": vector,
            }
        )

    result = await client.update_batch_point_vectors(batch_updates)
    assert result["success"] == 3, "All 3 updates should succeed"

    # Search with "Action Hero" query
    query_vector = await embedding_manager.text_processor.encode_text("Action Hero")
    assert query_vector is not None
    search_results = await client.search_single_vector(
        vector_name="text_vector",
        vector_data=query_vector,
        limit=20,  # Increased limit to ensure we find our test anime
    )

    # Verify our test anime are in results
    result_ids = [hit["id"] for hit in search_results]
    assert anime_list[0].anime.id in result_ids, (
        "Action Hero Adventure should be in results"
    )
    assert anime_list[1].anime.id in result_ids, (
        "Action Hero Story should be in results"
    )

    # Get scores for action anime
    similar1_result = next(
        r for r in search_results if r["id"] == anime_list[0].anime.id
    )
    similar2_result = next(
        r for r in search_results if r["id"] == anime_list[1].anime.id
    )

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
    client: QdrantClient, embedding_manager
):
    """Test handling when field_mapper extraction returns empty/minimal data."""
    # Create anime with minimal data
    minimal_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="M",  # Very short title
        genres=[],  # Empty genre
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [minimal_anime], batch_size=1)

    # Try to extract and update vectors for minimal anime
    content = embedding_manager.field_mapper.extract_anime_text(minimal_anime.anime)

    # Even with minimal data, extraction should return something
    assert content is not None, "Extraction should not return None"

    # Encode and update
    vector = await embedding_manager.text_processor.encode_text(content)
    assert vector is not None

    result = await client.update_batch_point_vectors(
        [
            {
                "point_id": minimal_anime.anime.id,
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
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(5)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=len(test_anime))

    batch_updates = [
        # Invalid vector name
        {
            "point_id": test_anime[0].anime.id,
            "vector_name": "invalid_name",
            "vector_data": [0.1] * 1024,
        },
        # Wrong dimension
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": [0.1] * 512,
        },
        # Invalid data type
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": "not a vector",
        },
        # None value
        {
            "point_id": test_anime[3].anime.id,
            "vector_name": "text_vector",
            "vector_data": None,
        },
        # Empty vector
        {
            "point_id": test_anime[4].anime.id,
            "vector_name": "text_vector",
            "vector_data": [],
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    assert result["failed"] == 5, "All 5 should fail"
    assert len(result["results"]) == 5, "Should have 5 detailed results"

    # Verify each has specific error message
    for r in result["results"]:
        assert not r["success"], "All should fail"
        assert "error" in r, "Must have error field"
        assert len(r["error"]) > 0, "Error message must not be empty"
        assert "point_id" in r, "Must have point_id"
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
    client: QdrantClient, embedding_manager
):
    """Test that statistics are correctly aggregated across multiple sequential batches."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(30)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=30)

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
                vector_opt = await embedding_manager.text_processor.encode_text(content)
                assert vector_opt is not None
                vector = vector_opt

            batch_updates.append(
                {
                    "point_id": test_anime[i].anime.id,
                    "vector_name": "text_vector",
                    "vector_data": vector,
                }
            )

        result = await client.update_batch_point_vectors(batch_updates)

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
async def test_result_structure_completeness(client: QdrantClient, embedding_manager):
    """Test that result structure contains all required fields for both success and failure."""
    test_anime = build_anime_record(
        anime_id=str(uuid.uuid4()),
        title="Structure Test",
        genres=["Action"],
        year=2020,
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        sources=[],
    )

    await add_bare_anime_points(client, [test_anime], batch_size=1)

    content = embedding_manager.field_mapper.extract_anime_text(test_anime.anime)
    vector = await embedding_manager.text_processor.encode_text(content)
    assert vector is not None

    batch_updates = [
        # Success case
        {
            "point_id": test_anime.anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        },
        # Failure case
        {
            "point_id": test_anime.anime.id,
            "vector_name": "invalid_vector",
            "vector_data": vector,
        },
    ]

    result = await client.update_batch_point_vectors(batch_updates)

    # Check top-level structure
    assert "success" in result, "Must have success count"
    assert "failed" in result, "Must have failed count"
    assert "results" in result, "Must have results list"
    assert isinstance(result["success"], int), "success must be int"
    assert isinstance(result["failed"], int), "failed must be int"
    assert isinstance(result["results"], list), "results must be list"

    # Check success result structure
    success_result = next(r for r in result["results"] if r["success"])
    assert "point_id" in success_result, "Success must have point_id"
    assert "vector_name" in success_result, "Success must have vector_name"
    assert "success" in success_result, "Success must have success field"
    assert success_result["success"] is True, "success field must be True"
    assert "error" not in success_result or success_result.get("error") is None, (
        "Success should not have error"
    )

    # Check failure result structure
    failure_result = next(r for r in result["results"] if not r["success"])
    assert "point_id" in failure_result, "Failure must have point_id"
    assert "vector_name" in failure_result, "Failure must have vector_name"
    assert "success" in failure_result, "Failure must have success field"
    assert failure_result["success"] is False, "success field must be False"
    assert "error" in failure_result, "Failure must have error field"
    assert isinstance(failure_result["error"], str), "error must be string"
    assert len(failure_result["error"]) > 0, "error must not be empty"


@pytest.mark.asyncio
async def test_update_with_different_vector_combinations(
    client: QdrantClient, embedding_manager
):
    """Test various combinations of vector updates in single batch."""
    test_anime = [
        build_anime_record(
            anime_id=str(uuid.uuid4()),
            title=f"Combo Test {i}",
            genres=["Action"],
            year=2020,
            type=AnimeType.TV,
            status=AnimeStatus.FINISHED,
            sources=[],
        )
        for i in range(3)
    ]

    await add_bare_anime_points(client, test_anime, batch_size=3)

    batch_updates = []

    # Anime 0: Only text_vector
    content = embedding_manager.field_mapper.extract_anime_text(test_anime[0].anime)
    vector = await embedding_manager.text_processor.encode_text(content)
    assert vector is not None
    batch_updates.append(
        {
            "point_id": test_anime[0].anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        }
    )

    # Anime 1: text_vector + image_vector
    content = embedding_manager.field_mapper.extract_anime_text(test_anime[1].anime)
    vector = await embedding_manager.text_processor.encode_text(content)
    assert vector is not None
    batch_updates.append(
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        }
    )
    batch_updates.append(
        {
            "point_id": test_anime[1].anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],  # Multivector format
        }
    )

    # Anime 2: text_vector + image_vector + invalid
    content = embedding_manager.field_mapper.extract_anime_text(test_anime[2].anime)
    vector = await embedding_manager.text_processor.encode_text(content)
    assert vector is not None
    batch_updates.append(
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "text_vector",
            "vector_data": vector,
        }
    )
    batch_updates.append(
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "image_vector",
            "vector_data": [[0.1] * 768],  # Multivector format
        }
    )
    batch_updates.append(
        {
            "point_id": test_anime[2].anime.id,
            "vector_name": "invalid_vector",
            "vector_data": vector,
        }
    )

    result = await client.update_batch_point_vectors(batch_updates)

    # Total: 1 (Anime 0) + 2 (Anime 1) + 2 (Anime 2) = 5 successful vector updates
    assert result["success"] == 5, "All 5 valid updates should succeed"
    assert result["failed"] == 1, "1 invalid vector name should fail"

    # Verify per-anime breakdown
    anime_results = {}
    for r in result["results"]:
        anime_id = r["point_id"]
        if anime_id not in anime_results:
            anime_results[anime_id] = 0
        if r["success"]:
            anime_results[anime_id] += 1

    assert anime_results[test_anime[0].anime.id] == 1, "Anime 0 should have 1 update"
    assert anime_results[test_anime[1].anime.id] == 2, "Anime 1 should have 2 updates"
    assert anime_results[test_anime[2].anime.id] == 2, (
        "Anime 2 should have 2 successful updates"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
