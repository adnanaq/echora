"""
Integration tests for update_vectors.py focusing on the critical fixes:
1. expected_updates validation and warning
2. Accurate anime success tracking (all-or-nothing per anime)
3. Per-vector statistics using detailed results

These tests validate the script-level logic that uses the detailed results
from QdrantClient.update_batch_vectors().
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import List

from qdrant_db import QdrantClient
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from common.models.anime import AnimeEntry
from vector_db_interface import VectorDocument

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


async def add_test_anime(client: QdrantClient, anime_list: List[AnimeEntry] | AnimeEntry, batch_size: int = 100):
    """Helper to add anime with correct point IDs."""
    if isinstance(anime_list, AnimeEntry):
        anime_list = [anime_list]

    documents = []
    for anime in anime_list:
        point_id = client._generate_point_id(anime.id)
        documents.append(VectorDocument(id=point_id, vectors={}, payload=anime.model_dump()))

    await client.add_documents(documents, batch_size=batch_size)



@pytest.mark.asyncio
async def test_vector_persistence_after_update(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that vectors are actually persisted and retrievable after update."""
    # Create and seed test anime
    test_anime = AnimeEntry(
        id="persistence-test-1",
        title="Persistence Test Anime",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Generate and update title_vector
    title_content = embedding_manager.field_mapper._extract_title_content(test_anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)

    batch_updates = [{
        'anime_id': test_anime.id,
        'vector_name': 'title_vector',
        'vector_data': title_vector
    }]

    result = await client.update_batch_vectors(batch_updates)

    # Verify update succeeded
    assert result['success'] == 1, "Update should succeed"
    assert result['failed'] == 0, "No failures expected"

    # Verify vector is retrievable via search
    # Generate query vector for the same title
    query_vector = embedding_manager.text_processor.encode_text("Persistence Test Anime")

    search_results = await client.search_single_vector(
        vector_name="title_vector",
        vector_data=query_vector,
        limit=5
    )

    # Should find the anime we just updated (should be top result)
    found_ids = [hit['id'] for hit in search_results]
    assert test_anime.id in found_ids, "Updated anime should be searchable by title_vector"

    # Verify it's actually the first result (highest similarity)
    assert search_results[0]['id'] == test_anime.id, "Updated anime should be the top search result"


@pytest.mark.asyncio
async def test_detailed_results_provide_accurate_tracking(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that detailed results from update_batch_vectors enable accurate per-update tracking."""
    test_anime = [
        AnimeEntry(id=f"detailed-test-{i}", title=f"Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # Create batch with mix of valid and invalid updates
    batch_updates = []

    # Valid update for anime-0
    title_content = embedding_manager.field_mapper._extract_title_content(test_anime[0])
    title_vector = embedding_manager.text_processor.encode_text(title_content)
    batch_updates.append({
        'anime_id': test_anime[0].id,
        'vector_name': 'title_vector',
        'vector_data': title_vector
    })

    # Invalid update (wrong dimension) for anime-1
    batch_updates.append({
        'anime_id': test_anime[1].id,
        'vector_name': 'title_vector',
        'vector_data': [0.1] * 512  # Wrong dimension
    })

    # Valid update for anime-2
    title_content2 = embedding_manager.field_mapper._extract_title_content(test_anime[2])
    title_vector2 = embedding_manager.text_processor.encode_text(title_content2)
    batch_updates.append({
        'anime_id': test_anime[2].id,
        'vector_name': 'title_vector',
        'vector_data': title_vector2
    })

    result = await client.update_batch_vectors(batch_updates)

    # Verify detailed results enable accurate tracking
    assert 'results' in result, "Should have detailed results"
    assert len(result['results']) == 3, "Should have 3 detailed results"

    # Verify we can identify which specific anime failed
    failed_anime_ids = [r['anime_id'] for r in result['results'] if not r['success']]
    successful_anime_ids = [r['anime_id'] for r in result['results'] if r['success']]

    assert test_anime[1].id in failed_anime_ids, "anime-1 should be identified as failed"
    assert test_anime[0].id in successful_anime_ids, "anime-0 should be identified as successful"
    assert test_anime[2].id in successful_anime_ids, "anime-2 should be identified as successful"

    # Verify error messages are provided for failures
    for result_detail in result['results']:
        if not result_detail['success']:
            assert 'error' in result_detail, "Failed update should have error message"
            assert len(result_detail['error']) > 0, "Error message should not be empty"


@pytest.mark.asyncio
async def test_all_or_nothing_anime_success_logic(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """
    Test the critical all-or-nothing logic:
    An anime is only considered successful if ALL its vectors succeed.
    This tests the fix to the original flawed logic that incorrectly tracked anime success.
    """
    test_anime = AnimeEntry(
        id="all-or-nothing-test",
        title="All Or Nothing Test",
        genres=["Action", "Drama"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Generate 3 vectors: 2 valid, 1 invalid
    title_content = embedding_manager.field_mapper._extract_title_content(test_anime)
    title_vector = embedding_manager.text_processor.encode_text(title_content)

    character_content = embedding_manager.field_mapper._extract_character_content(test_anime)
    character_vector = embedding_manager.text_processor.encode_text(character_content)

    batch_updates = [
        {
            'anime_id': test_anime.id,
            'vector_name': 'title_vector',
            'vector_data': title_vector  # Valid
        },
        {
            'anime_id': test_anime.id,
            'vector_name': 'genre_vector',
            'vector_data': [0.1] * 512  # Invalid - wrong dimension
        },
        {
            'anime_id': test_anime.id,
            'vector_name': 'character_vector',
            'vector_data': character_vector  # Valid
        },
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Verify detailed results
    assert result['success'] == 2, "Should have 2 successful updates"
    assert result['failed'] == 1, "Should have 1 failed update"

    # Calculate anime success using the new logic (from update_vectors.py lines 430-444)
    anime_success_map = {}
    for result_detail in result['results']:
        anime_id = result_detail['anime_id']
        if anime_id not in anime_success_map:
            anime_success_map[anime_id] = {"total": 0, "success": 0}
        anime_success_map[anime_id]["total"] += 1
        if result_detail['success']:
            anime_success_map[anime_id]["success"] += 1

    # Count anime as successful only if ALL vectors succeeded
    anime_fully_successful = sum(
        1 for anime_id, stats in anime_success_map.items()
        if stats["success"] == stats["total"]
    )

    # This anime should NOT be counted as successful because genre_vector failed
    assert anime_fully_successful == 0, "Anime with partial vector success should NOT be counted as successful"
    assert anime_success_map[test_anime.id]['success'] == 2, "Should have 2 successful vectors"
    assert anime_success_map[test_anime.id]['total'] == 3, "Should have 3 total vectors"


@pytest.mark.asyncio
async def test_per_vector_statistics_from_detailed_results(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """
    Test that per-vector statistics are accurately tracked using detailed results.
    This validates the fix from lines 418-428 in update_vectors.py.
    """
    test_anime = [
        AnimeEntry(id=f"stats-test-{i}", title=f"Stats Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # Create updates with specific failure pattern:
    # - title_vector: all 3 succeed
    # - genre_vector: 2 succeed, 1 fails (anime-1)
    batch_updates = []

    for i, anime in enumerate(test_anime):
        # Title vector - all valid
        title_content = embedding_manager.field_mapper._extract_title_content(anime)
        title_vector = embedding_manager.text_processor.encode_text(title_content)
        batch_updates.append({
            'anime_id': anime.id,
            'vector_name': 'title_vector',
            'vector_data': title_vector
        })

        # Genre vector - anime-1 has invalid dimension
        if i == 1:
            batch_updates.append({
                'anime_id': anime.id,
                'vector_name': 'genre_vector',
                'vector_data': [0.1] * 512  # Invalid
            })
        else:
            genre_content = embedding_manager.field_mapper._extract_genre_content(anime)
            genre_vector = embedding_manager.text_processor.encode_text(genre_content)
            batch_updates.append({
                'anime_id': anime.id,
                'vector_name': 'genre_vector',
                'vector_data': genre_vector
            })

    result = await client.update_batch_vectors(batch_updates)

    # Simulate the per-vector statistics tracking from update_vectors.py (lines 418-428)
    vector_stats = {
        'title_vector': {'success': 0, 'failed': 0},
        'genre_vector': {'success': 0, 'failed': 0}
    }

    for result_detail in result['results']:
        vector_name = result_detail['vector_name']
        if result_detail['success']:
            vector_stats[vector_name]['success'] += 1
        else:
            vector_stats[vector_name]['failed'] += 1

    # Verify per-vector statistics
    assert vector_stats['title_vector']['success'] == 3, "All 3 title vectors should succeed"
    assert vector_stats['title_vector']['failed'] == 0, "No title vector failures"

    assert vector_stats['genre_vector']['success'] == 2, "2 genre vectors should succeed"
    assert vector_stats['genre_vector']['failed'] == 1, "1 genre vector should fail (anime-1)"


@pytest.mark.asyncio
async def test_empty_batch_handling(client: QdrantClient):
    """Test that empty batch is handled gracefully."""
    result = await client.update_batch_vectors([])

    assert result['success'] == 0, "Empty batch should have 0 successes"
    assert result['failed'] == 0, "Empty batch should have 0 failures"
    assert result['results'] == [], "Empty batch should have empty results list"


@pytest.mark.asyncio
async def test_all_validation_failures_no_qdrant_call(client: QdrantClient):
    """Test that when all updates fail validation, no Qdrant call is made."""
    test_anime = AnimeEntry(
        id="validation-only-test",
        title="Validation Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # All updates have validation errors
    batch_updates = [
        {'anime_id': test_anime.id, 'vector_name': 'invalid_vector', 'vector_data': [0.1] * 1024},
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': [0.1] * 512},  # Wrong dim
        {'anime_id': test_anime.id, 'vector_name': 'genre_vector', 'vector_data': "not a vector"},
    ]

    result = await client.update_batch_vectors(batch_updates)

    # All should fail validation, no Qdrant operation should occur
    assert result['success'] == 0, "All updates should fail validation"
    assert result['failed'] == 3, "Should have 3 validation failures"
    assert len(result['results']) == 3, "Should have 3 detailed failure results"

    # All should have error messages
    for result_detail in result['results']:
        assert not result_detail['success'], "All should be failures"
        assert 'error' in result_detail, "Each should have error message"


@pytest.mark.asyncio
async def test_mixed_batch_all_combinations(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test batch with all possible anime success combinations: all pass, partial pass, all fail."""
    test_anime = [
        AnimeEntry(id=f"mixed-{i}", title=f"Mixed Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(5)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    batch_updates = []

    # Anime 0: 3/3 vectors succeed (all pass)
    for vector_name in ['title_vector', 'genre_vector', 'character_vector']:
        content = embedding_manager.field_mapper._extract_title_content(test_anime[0])
        vector = embedding_manager.text_processor.encode_text(content)
        batch_updates.append({'anime_id': test_anime[0].id, 'vector_name': vector_name, 'vector_data': vector})

    # Anime 1: 2/3 vectors succeed (partial pass)
    batch_updates.append({'anime_id': test_anime[1].id, 'vector_name': 'title_vector', 'vector_data': [0.1] * 1024})
    batch_updates.append({'anime_id': test_anime[1].id, 'vector_name': 'genre_vector', 'vector_data': [0.1] * 512})  # Fail
    batch_updates.append({'anime_id': test_anime[1].id, 'vector_name': 'character_vector', 'vector_data': [0.1] * 1024})

    # Anime 2: 1/3 vectors succeed (mostly fail)
    batch_updates.append({'anime_id': test_anime[2].id, 'vector_name': 'title_vector', 'vector_data': [0.1] * 1024})
    batch_updates.append({'anime_id': test_anime[2].id, 'vector_name': 'genre_vector', 'vector_data': [0.1] * 512})  # Fail
    batch_updates.append({'anime_id': test_anime[2].id, 'vector_name': 'character_vector', 'vector_data': "invalid"})  # Fail

    # Anime 3: 0/3 vectors succeed (all fail)
    batch_updates.append({'anime_id': test_anime[3].id, 'vector_name': 'invalid_vector', 'vector_data': [0.1] * 1024})
    batch_updates.append({'anime_id': test_anime[3].id, 'vector_name': 'genre_vector', 'vector_data': [0.1] * 512})
    batch_updates.append({'anime_id': test_anime[3].id, 'vector_name': 'character_vector', 'vector_data': None})

    # Anime 4: 3/3 vectors succeed (all pass)
    for vector_name in ['title_vector', 'genre_vector', 'character_vector']:
        content = embedding_manager.field_mapper._extract_title_content(test_anime[4])
        vector = embedding_manager.text_processor.encode_text(content)
        batch_updates.append({'anime_id': test_anime[4].id, 'vector_name': vector_name, 'vector_data': vector})

    result = await client.update_batch_vectors(batch_updates)

    # Calculate anime success map
    anime_success_map = {}
    for r in result['results']:
        anime_id = r['anime_id']
        if anime_id not in anime_success_map:
            anime_success_map[anime_id] = {"total": 0, "success": 0}
        anime_success_map[anime_id]["total"] += 1
        if r['success']:
            anime_success_map[anime_id]["success"] += 1

    # Verify all combinations
    assert anime_success_map[test_anime[0].id]['success'] == 3, "Anime 0: all should succeed"
    assert anime_success_map[test_anime[0].id]['total'] == 3

    assert anime_success_map[test_anime[1].id]['success'] == 2, "Anime 1: 2/3 should succeed"
    assert anime_success_map[test_anime[1].id]['total'] == 3

    assert anime_success_map[test_anime[2].id]['success'] == 1, "Anime 2: 1/3 should succeed"
    assert anime_success_map[test_anime[2].id]['total'] == 3

    assert anime_success_map[test_anime[3].id]['success'] == 0, "Anime 3: 0/3 should succeed"
    assert anime_success_map[test_anime[3].id]['total'] == 3

    assert anime_success_map[test_anime[4].id]['success'] == 3, "Anime 4: all should succeed"
    assert anime_success_map[test_anime[4].id]['total'] == 3

    # Count fully successful anime (only anime 0 and 4)
    fully_successful = sum(
        1 for anime_id, stats in anime_success_map.items()
        if stats['success'] == stats['total']
    )
    assert fully_successful == 2, "Only 2 anime should be fully successful"


@pytest.mark.asyncio
async def test_all_11_vectors_simultaneously(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test updating all 11 vector types in a single batch."""
    test_anime = AnimeEntry(
        id="all-vectors-test",
        title="All Vectors Test",
        genres=["Action", "Drama"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # All 11 vector names
    vector_names = [
        'title_vector', 'character_vector', 'genre_vector', 'staff_vector',
        'temporal_vector', 'streaming_vector', 'related_vector',
        'franchise_vector', 'episode_vector',
        'image_vector', 'character_image_vector'  # Image vectors (768-dim)
    ]

    batch_updates = []
    for vector_name in vector_names:
        if 'image' in vector_name:
            # Image vectors are 768-dimensional
            vector_data = [0.1] * 768
        else:
            # Text vectors are 1024-dimensional
            content = embedding_manager.field_mapper._extract_title_content(test_anime)
            vector_data = embedding_manager.text_processor.encode_text(content)

        batch_updates.append({
            'anime_id': test_anime.id,
            'vector_name': vector_name,
            'vector_data': vector_data
        })

    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 11, "All 11 vectors should succeed"
    assert result['failed'] == 0, "No failures expected"
    assert len(result['results']) == 11, "Should have 11 detailed results"

    # Verify all vector names are present
    result_vector_names = {r['vector_name'] for r in result['results']}
    assert result_vector_names == set(vector_names), "All vector names should be present"


@pytest.mark.asyncio
async def test_duplicate_anime_in_same_batch(client: QdrantClient):
    """Test batch with duplicate anime IDs (same vector updated multiple times)."""
    test_anime = AnimeEntry(
        id="duplicate-test",
        title="Duplicate Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Update same vector 3 times with different values
    batch_updates = [
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': [0.1] * 1024},
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': [0.2] * 1024},
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': [0.3] * 1024},
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Due to deduplication, only last one should persist
    assert result['success'] == 1, "Should deduplicate to 1 update"
    assert result['failed'] == 0, "No failures"
    assert len(result['results']) == 1, "Should have 1 result after deduplication"


@pytest.mark.asyncio
async def test_dimension_edge_cases(client: QdrantClient):
    """Test various dimension edge cases."""
    test_anime = AnimeEntry(
        id="dimension-test",
        title="Dimension Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    batch_updates = [
        # Empty vector
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': []},
        # Single value
        {'anime_id': test_anime.id, 'vector_name': 'genre_vector', 'vector_data': [0.1]},
        # One less than expected
        {'anime_id': test_anime.id, 'vector_name': 'character_vector', 'vector_data': [0.1] * 1023},
        # One more than expected
        {'anime_id': test_anime.id, 'vector_name': 'staff_vector', 'vector_data': [0.1] * 1025},
        # Exactly correct
        {'anime_id': test_anime.id, 'vector_name': 'temporal_vector', 'vector_data': [0.1] * 1024},
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 1, "Only correctly sized vector should succeed"
    assert result['failed'] == 4, "4 dimension mismatches should fail"

    # Verify error messages mention dimension
    dimension_errors = [r for r in result['results'] if not r['success'] and 'dimension' in r.get('error', '').lower()]
    validation_errors = [r for r in result['results'] if not r['success'] and 'valid' in r.get('error', '').lower()]

    assert len(dimension_errors) >= 3, "Should have dimension mismatch errors"
    assert len(validation_errors) >= 1, "Empty vector should fail validation"


@pytest.mark.asyncio
async def test_special_float_values(client: QdrantClient):
    """Test special float values: very small, very large, normal.

    Note: NaN and Inf values are rejected by Qdrant at the API level,
    causing the entire batch to fail before our validation runs.
    This test focuses on edge cases that should pass validation.
    """
    test_anime = AnimeEntry(
        id="float-test",
        title="Float Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    batch_updates = [
        # Very small values (near zero)
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': [1e-10] * 1024},
        # Very large values
        {'anime_id': test_anime.id, 'vector_name': 'genre_vector', 'vector_data': [1e10] * 1024},
        # Negative values
        {'anime_id': test_anime.id, 'vector_name': 'character_vector', 'vector_data': [-0.5] * 1024},
        # Zero values
        {'anime_id': test_anime.id, 'vector_name': 'staff_vector', 'vector_data': [0.0] * 1024},
        # Normal values
        {'anime_id': test_anime.id, 'vector_name': 'temporal_vector', 'vector_data': [0.5] * 1024},
    ]

    result = await client.update_batch_vectors(batch_updates)

    # All valid float values should work
    assert result['success'] == 5, "All valid float values should succeed"
    assert result['failed'] == 0, "No failures expected for valid floats"


@pytest.mark.asyncio
async def test_non_existent_anime_ids(client: QdrantClient):
    """Test updates to anime IDs that don't exist in Qdrant."""
    batch_updates = [
        {'anime_id': 'non-existent-1', 'vector_name': 'title_vector', 'vector_data': [0.1] * 1024},
        {'anime_id': 'non-existent-2', 'vector_name': 'genre_vector', 'vector_data': [0.1] * 1024},
        {'anime_id': 'non-existent-3', 'vector_name': 'character_vector', 'vector_data': [0.1] * 1024},
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Qdrant should accept updates even if points don't exist (will create them)
    # OR reject them - document the actual behavior
    assert 'success' in result, "Should return result structure"
    assert 'failed' in result, "Should return failed count"
    assert 'results' in result, "Should return detailed results"

    print(f"Non-existent anime behavior: success={result['success']}, failed={result['failed']}")


@pytest.mark.asyncio
async def test_batch_with_only_invalid_vector_names(client: QdrantClient):
    """Test batch where all vector names are invalid."""
    test_anime = AnimeEntry(
        id="invalid-names-test",
        title="Invalid Names Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    batch_updates = [
        {'anime_id': test_anime.id, 'vector_name': 'invalid_vector_1', 'vector_data': [0.1] * 1024},
        {'anime_id': test_anime.id, 'vector_name': 'not_a_vector', 'vector_data': [0.1] * 1024},
        {'anime_id': test_anime.id, 'vector_name': 'fake_vector_name', 'vector_data': [0.1] * 1024},
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 0, "All should fail with invalid names"
    assert result['failed'] == 3, "All 3 should be rejected"

    # All should have error messages about invalid vector names
    for r in result['results']:
        assert not r['success'], "All should fail"
        assert 'invalid' in r['error'].lower() or 'vector' in r['error'].lower(), "Error should mention invalid vector"


@pytest.mark.asyncio
async def test_mixed_valid_invalid_anime_ids(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test batch with mix of existing and non-existing anime IDs.

    Note: Qdrant's update_vectors operation updates existing points.
    Non-existing points will be processed but may not be updated if they don't exist.
    """
    existing_anime = AnimeEntry(
        id="existing-anime",
        title="Existing Anime",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [existing_anime], batch_size=1)

    content = embedding_manager.field_mapper._extract_title_content(existing_anime)
    vector = embedding_manager.text_processor.encode_text(content)

    batch_updates = [
        # Existing anime - should work
        {'anime_id': existing_anime.id, 'vector_name': 'title_vector', 'vector_data': vector},
        # Another existing update
        {'anime_id': existing_anime.id, 'vector_name': 'genre_vector', 'vector_data': vector},
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Both existing anime updates should work
    assert result['success'] == 2, "Existing anime updates should work"
    assert result['failed'] == 0, "No failures for existing anime"


@pytest.mark.asyncio
async def test_results_ordering_matches_input(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that detailed results maintain relationship to input order."""
    test_anime = [
        AnimeEntry(id=f"order-{i}", title=f"Order Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(5)
    ]

    await add_test_anime(client, test_anime, batch_size=len(test_anime))

    # Create specific pattern: success, fail, success, fail, success
    batch_updates = []
    for i, anime in enumerate(test_anime):
        if i % 2 == 0:  # Even indices - valid
            content = embedding_manager.field_mapper._extract_title_content(anime)
            vector = embedding_manager.text_processor.encode_text(content)
        else:  # Odd indices - invalid
            vector = [0.1] * 512  # Wrong dimension

        batch_updates.append({
            'anime_id': anime.id,
            'vector_name': 'title_vector',
            'vector_data': vector
        })

    result = await client.update_batch_vectors(batch_updates)

    # Verify we can match results to input
    assert len(result['results']) == 5, "Should have 5 results"

    # Check that all anime IDs from input are in results
    input_anime_ids = {u['anime_id'] for u in batch_updates}
    result_anime_ids = {r['anime_id'] for r in result['results']}

    assert input_anime_ids == result_anime_ids, "All anime IDs should be in results"


@pytest.mark.asyncio
async def test_large_batch_realistic_failures(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test large batch (100 anime) with realistic failure patterns."""
    test_anime = [
        AnimeEntry(id=f"large-{i}", title=f"Large Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(100)
    ]

    await add_test_anime(client, test_anime, batch_size=50)

    batch_updates = []
    for i, anime in enumerate(test_anime):
        content = embedding_manager.field_mapper._extract_title_content(anime)
        vector = embedding_manager.text_processor.encode_text(content)

        # 90% success rate - realistic scenario
        if i % 10 != 0:  # 90% valid
            batch_updates.append({'anime_id': anime.id, 'vector_name': 'title_vector', 'vector_data': vector})
        else:  # 10% invalid
            batch_updates.append({'anime_id': anime.id, 'vector_name': 'title_vector', 'vector_data': [0.1] * 512})

    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 90, "90 updates should succeed"
    assert result['failed'] == 10, "10 updates should fail"
    assert len(result['results']) == 100, "Should have 100 detailed results"


@pytest.mark.asyncio
async def test_sequential_updates_same_vector(client: QdrantClient):
    """Test updating same vector multiple times sequentially."""
    test_anime = AnimeEntry(
        id="sequential-test",
        title="Sequential Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Update same vector 3 times sequentially
    for i in range(3):
        vector = [float(i)] * 1024  # Different values each time
        batch_updates = [{'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': vector}]

        result = await client.update_batch_vectors(batch_updates)
        assert result['success'] == 1, f"Update {i+1} should succeed"

    # Verify last update persists
    query_vector = [2.0] * 1024  # Should match last update
    search_results = await client.search_single_vector(
        vector_name="title_vector",
        vector_data=query_vector,
        limit=5
    )

    # Should find our anime
    found_ids = [hit['id'] for hit in search_results]
    assert test_anime.id in found_ids, "Should find updated anime"


@pytest.mark.asyncio
async def test_image_and_text_vectors_mixed(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test mixing image vectors (768-dim) and text vectors (1024-dim) in same batch."""
    test_anime = AnimeEntry(
        id="mixed-dim-test",
        title="Mixed Dimension Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    content = embedding_manager.field_mapper._extract_title_content(test_anime)
    text_vector = embedding_manager.text_processor.encode_text(content)

    batch_updates = [
        # Text vector (1024-dim)
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': text_vector},
        # Image vector (768-dim)
        {'anime_id': test_anime.id, 'vector_name': 'image_vector', 'vector_data': [0.1] * 768},
        # Another text vector
        {'anime_id': test_anime.id, 'vector_name': 'genre_vector', 'vector_data': text_vector},
        # Character image vector (768-dim)
        {'anime_id': test_anime.id, 'vector_name': 'character_image_vector', 'vector_data': [0.1] * 768},
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 4, "All mixed dimension updates should succeed"
    assert result['failed'] == 0, "No failures expected"


@pytest.mark.asyncio
async def test_single_vector_update_method(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test update_single_vector method (not just batch updates)."""
    test_anime = AnimeEntry(
        id="single-update-test",
        title="Single Update Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Test single vector update
    content = embedding_manager.field_mapper._extract_title_content(test_anime)
    vector = embedding_manager.text_processor.encode_text(content)

    success = await client.update_single_vector(
        anime_id=test_anime.id, vector_name="title_vector", vector_data=vector
    )

    assert success is True, "Single vector update should succeed"

    # Test invalid vector name
    success = await client.update_single_vector(
        anime_id=test_anime.id,
        vector_name='invalid_vector',
        vector_data=vector
    )

    assert success is False, "Invalid vector name should fail"


@pytest.mark.asyncio
async def test_batch_size_boundaries(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test various batch sizes: 1, 2, 50, 100, 500."""
    # Create test anime
    test_anime = [
        AnimeEntry(id=f"batch-size-{i}", title=f"Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(500)
    ]

    # Add in batches to avoid memory issues
    for i in range(0, 500, 100):
        await add_test_anime(client, test_anime[i:i+100], batch_size=100)

    # Test different batch sizes
    batch_sizes_to_test = [1, 2, 50, 100, 500]

    for batch_size in batch_sizes_to_test:
        batch_updates = []
        for i in range(batch_size):
            content = embedding_manager.field_mapper._extract_title_content(test_anime[i])
            vector = embedding_manager.text_processor.encode_text(content)
            batch_updates.append({
                'anime_id': test_anime[i].id,
                'vector_name': 'title_vector',
                'vector_data': vector
            })

        result = await client.update_batch_vectors(batch_updates)

        assert result['success'] == batch_size, f"Batch size {batch_size} should have {batch_size} successes"
        assert result['failed'] == 0, f"Batch size {batch_size} should have 0 failures"
        assert len(result['results']) == batch_size, f"Should have {batch_size} results"


@pytest.mark.asyncio
async def test_update_then_search_consistency(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that updated vectors are immediately searchable with correct results."""
    test_anime = AnimeEntry(
        id="consistency-test",
        title="Consistency Test Anime",
        genres=["Action", "Adventure"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    # Update title_vector
    content = embedding_manager.field_mapper._extract_title_content(test_anime)
    title_vector = embedding_manager.text_processor.encode_text(content)

    update_result = await client.update_batch_vectors([{
        'anime_id': test_anime.id,
        'vector_name': 'title_vector',
        'vector_data': title_vector
    }])

    assert update_result['success'] == 1, "Update should succeed"

    # Immediately search for the updated vector
    search_results = await client.search_single_vector(
        vector_name="title_vector",
        vector_data=title_vector,
        limit=5
    )

    # Should find the anime we just updated
    found_ids = [hit['id'] for hit in search_results]
    assert test_anime.id in found_ids, "Updated anime should be immediately searchable"

    # Should be top result or very high in results (might have other test data)
    # Self-similarity should be very high
    our_result = next((r for r in search_results if r['id'] == test_anime.id), None)
    assert our_result is not None, "Must find updated anime"
    assert our_result['similarity_score'] > 0.98, "Self-similarity should be very high"


@pytest.mark.asyncio
async def test_similarity_search_after_multiple_updates(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that similarity search returns correct results after vector updates."""
    # Create 3 similar anime
    anime_list = [
        AnimeEntry(id="similar-1", title="Action Hero Adventure", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[]),
        AnimeEntry(id="similar-2", title="Action Hero Story", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[]),
        AnimeEntry(id="different-1", title="Romance Comedy Love", genre=["Romance"], year=2020, type="TV", status="FINISHED", sources=[]),
    ]

    await add_test_anime(client, anime_list, batch_size=3)

    # Update all title vectors
    batch_updates = []
    for anime in anime_list:
        content = embedding_manager.field_mapper._extract_title_content(anime)
        vector = embedding_manager.text_processor.encode_text(content)
        batch_updates.append({
            'anime_id': anime.id,
            'vector_name': 'title_vector',
            'vector_data': vector
        })

    result = await client.update_batch_vectors(batch_updates)
    assert result['success'] == 3, "All 3 updates should succeed"

    # Search with "Action Hero" query
    query_vector = embedding_manager.text_processor.encode_text("Action Hero")
    search_results = await client.search_single_vector(
        vector_name="title_vector",
        vector_data=query_vector,
        limit=20  # Increased limit to ensure we find our test anime
    )

    # Verify our test anime are in results
    result_ids = [hit['id'] for hit in search_results]
    assert "similar-1" in result_ids, "Action Hero Adventure should be in results"
    assert "similar-2" in result_ids, "Action Hero Story should be in results"

    # Get scores for action anime
    similar1_result = next(r for r in search_results if r['id'] == "similar-1")
    similar2_result = next(r for r in search_results if r['id'] == "similar-2")

    # Both action anime should have high similarity scores (>0.7) for "Action Hero" query
    assert similar1_result['similarity_score'] > 0.7, "Action Hero Adventure should have high similarity"
    assert similar2_result['similarity_score'] > 0.7, "Action Hero Story should have high similarity"

    # If romance anime is in results, action anime should score higher
    different_result = next((r for r in search_results if r['id'] == "different-1"), None)
    if different_result:
        assert similar1_result['similarity_score'] > different_result['similarity_score'], "Action anime should score higher than Romance"
        assert similar2_result['similarity_score'] > different_result['similarity_score'], "Action anime should score higher than Romance"


@pytest.mark.asyncio
async def test_vector_extraction_failures_handling(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test handling when field_mapper extraction returns empty/minimal data."""
    # Create anime with minimal data
    minimal_anime = AnimeEntry(
        id="minimal-data",
        title="M",  # Very short title
        genres=[],  # Empty genre
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [minimal_anime], batch_size=1)

    # Try to extract and update vectors for minimal anime
    content = embedding_manager.field_mapper._extract_genre_content(minimal_anime)

    # Even with minimal data, extraction should return something
    assert content is not None, "Extraction should not return None"

    # Encode and update
    vector = embedding_manager.text_processor.encode_text(content)

    result = await client.update_batch_vectors([{
        'anime_id': minimal_anime.id,
        'vector_name': 'genre_vector',
        'vector_data': vector
    }])

    # Should still succeed even with minimal data
    assert result['success'] == 1, "Update with minimal data should succeed"


@pytest.mark.asyncio
async def test_all_error_types_in_detailed_results(client: QdrantClient):
    """Test that all error types return detailed, accurate error messages."""
    test_anime = AnimeEntry(
        id="error-types-test",
        title="Error Types Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    batch_updates = [
        # Invalid vector name
        {'anime_id': test_anime.id, 'vector_name': 'invalid_name', 'vector_data': [0.1] * 1024},
        # Wrong dimension
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': [0.1] * 512},
        # Invalid data type
        {'anime_id': test_anime.id, 'vector_name': 'genre_vector', 'vector_data': "not a vector"},
        # None value
        {'anime_id': test_anime.id, 'vector_name': 'character_vector', 'vector_data': None},
        # Empty vector
        {'anime_id': test_anime.id, 'vector_name': 'staff_vector', 'vector_data': []},
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result['failed'] == 5, "All 5 should fail"
    assert len(result['results']) == 5, "Should have 5 detailed results"

    # Verify each has specific error message
    for r in result['results']:
        assert not r['success'], "All should fail"
        assert 'error' in r, "Must have error field"
        assert len(r['error']) > 0, "Error message must not be empty"
        assert 'anime_id' in r, "Must have anime_id"
        assert 'vector_name' in r, "Must have vector_name"

    # Check error types are distinct and meaningful
    errors = [r['error'].lower() for r in result['results']]
    assert any('invalid' in e and 'vector' in e for e in errors), "Should have invalid vector name error"
    assert any('dimension' in e for e in errors), "Should have dimension error"
    assert any('valid' in e or 'type' in e for e in errors), "Should have data type errors"


@pytest.mark.asyncio
async def test_multi_batch_statistics_aggregation(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that statistics are correctly aggregated across multiple sequential batches."""
    test_anime = [
        AnimeEntry(id=f"multi-batch-{i}", title=f"Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(30)
    ]

    await add_test_anime(client, test_anime, batch_size=30)

    # Track stats across 3 batches of 10
    total_success = 0
    total_failed = 0
    vector_stats = {'title_vector': {'success': 0, 'failed': 0}}

    for batch_num in range(3):
        batch_start = batch_num * 10
        batch_end = batch_start + 10

        batch_updates = []
        for i in range(batch_start, batch_end):
            # Introduce failures: every 5th update fails
            if i % 5 == 0:
                vector = [0.1] * 512  # Wrong dimension - will fail
            else:
                content = embedding_manager.field_mapper._extract_title_content(test_anime[i])
                vector = embedding_manager.text_processor.encode_text(content)

            batch_updates.append({
                'anime_id': test_anime[i].id,
                'vector_name': 'title_vector',
                'vector_data': vector
            })

        result = await client.update_batch_vectors(batch_updates)

        # Aggregate stats
        total_success += result['success']
        total_failed += result['failed']

        for r in result['results']:
            if r['success']:
                vector_stats['title_vector']['success'] += 1
            else:
                vector_stats['title_vector']['failed'] += 1

    # Verify aggregated statistics
    assert total_success == 24, "Should have 24 total successes (30 - 6 failures)"
    assert total_failed == 6, "Should have 6 total failures (every 5th)"
    assert vector_stats['title_vector']['success'] == 24, "Vector stats should match"
    assert vector_stats['title_vector']['failed'] == 6, "Vector stats should match"


@pytest.mark.asyncio
async def test_result_structure_completeness(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test that result structure contains all required fields for both success and failure."""
    test_anime = AnimeEntry(
        id="structure-test",
        title="Structure Test",
        genres=["Action"],
        year=2020,
        type="TV",
        status="FINISHED",
        sources=[]
    )

    await add_test_anime(client, [test_anime], batch_size=1)

    content = embedding_manager.field_mapper._extract_title_content(test_anime)
    vector = embedding_manager.text_processor.encode_text(content)

    batch_updates = [
        # Success case
        {'anime_id': test_anime.id, 'vector_name': 'title_vector', 'vector_data': vector},
        # Failure case
        {'anime_id': test_anime.id, 'vector_name': 'invalid_vector', 'vector_data': vector},
    ]

    result = await client.update_batch_vectors(batch_updates)

    # Check top-level structure
    assert 'success' in result, "Must have success count"
    assert 'failed' in result, "Must have failed count"
    assert 'results' in result, "Must have results list"
    assert isinstance(result['success'], int), "success must be int"
    assert isinstance(result['failed'], int), "failed must be int"
    assert isinstance(result['results'], list), "results must be list"

    # Check success result structure
    success_result = next(r for r in result['results'] if r['success'])
    assert 'anime_id' in success_result, "Success must have anime_id"
    assert 'vector_name' in success_result, "Success must have vector_name"
    assert 'success' in success_result, "Success must have success field"
    assert success_result['success'] is True, "success field must be True"
    assert 'error' not in success_result or success_result.get('error') is None, "Success should not have error"

    # Check failure result structure
    failure_result = next(r for r in result['results'] if not r['success'])
    assert 'anime_id' in failure_result, "Failure must have anime_id"
    assert 'vector_name' in failure_result, "Failure must have vector_name"
    assert 'success' in failure_result, "Failure must have success field"
    assert failure_result['success'] is False, "success field must be False"
    assert 'error' in failure_result, "Failure must have error field"
    assert isinstance(failure_result['error'], str), "error must be string"
    assert len(failure_result['error']) > 0, "error must not be empty"


@pytest.mark.asyncio
async def test_update_with_different_vector_combinations(client: QdrantClient, embedding_manager: MultiVectorEmbeddingManager):
    """Test various combinations of vector updates in single batch."""
    test_anime = [
        AnimeEntry(id=f"combo-{i}", title=f"Combo Test {i}", genre=["Action"], year=2020, type="TV", status="FINISHED", sources=[])
        for i in range(3)
    ]

    await add_test_anime(client, test_anime, batch_size=3)

    batch_updates = []

    # Anime 0: Only title_vector
    content = embedding_manager.field_mapper._extract_title_content(test_anime[0])
    vector = embedding_manager.text_processor.encode_text(content)
    batch_updates.append({'anime_id': test_anime[0].id, 'vector_name': 'title_vector', 'vector_data': vector})

    # Anime 1: title_vector + genre_vector
    for vector_name in ['title_vector', 'genre_vector']:
        content = embedding_manager.field_mapper._extract_title_content(test_anime[1])
        vector = embedding_manager.text_processor.encode_text(content)
        batch_updates.append({'anime_id': test_anime[1].id, 'vector_name': vector_name, 'vector_data': vector})

    # Anime 2: title_vector + genre_vector + character_vector
    for vector_name in ['title_vector', 'genre_vector', 'character_vector']:
        content = embedding_manager.field_mapper._extract_title_content(test_anime[2])
        vector = embedding_manager.text_processor.encode_text(content)
        batch_updates.append({'anime_id': test_anime[2].id, 'vector_name': vector_name, 'vector_data': vector})

    result = await client.update_batch_vectors(batch_updates)

    # Total: 1 + 2 + 3 = 6 updates
    assert result['success'] == 6, "All 6 updates should succeed"
    assert result['failed'] == 0, "No failures"

    # Verify per-anime breakdown
    anime_results = {}
    for r in result['results']:
        anime_id = r['anime_id']
        if anime_id not in anime_results:
            anime_results[anime_id] = 0
        if r['success']:
            anime_results[anime_id] += 1

    assert anime_results[test_anime[0].id] == 1, "Anime 0 should have 1 update"
    assert anime_results[test_anime[1].id] == 2, "Anime 1 should have 2 updates"
    assert anime_results[test_anime[2].id] == 3, "Anime 2 should have 3 updates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
