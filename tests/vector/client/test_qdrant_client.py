#!/usr/bin/env python3
"""
Tests for QdrantClient update methods.

Tests the update_single_vector and update_batch_vectors methods
added for selective vector updates without full reindexing.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_settings
from src.models.anime import AnimeEntry
from src.vector.client.qdrant_client import QdrantClient


@pytest.fixture
def settings():
    """Get test settings."""
    return get_settings()


@pytest.fixture
def client(settings):
    """Create QdrantClient instance."""
    return QdrantClient(settings=settings)


@pytest.fixture
def test_anime_data():
    """Load test anime data from enriched database."""
    with open('./data/qdrant_storage/enriched_anime_database.json', 'r') as f:
        data = json.load(f)
    return data['data'][:3]  # Use first 3 anime for testing


@pytest.mark.asyncio
async def test_update_single_vector(client: QdrantClient, test_anime_data: List[Dict]):
    """Test updating a single vector for one anime."""
    # Get first anime
    anime_dict = test_anime_data[0]
    anime = AnimeEntry(**anime_dict)

    # Generate title vector
    title_content = client.embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = client.embedding_manager.text_processor.encode_text(title_content)

    # Update single vector
    success = await client.update_single_vector(
        anime_id=anime.id,
        vector_name='title_vector',
        vector_data=title_vector
    )

    assert success is True, "Single vector update should succeed"


@pytest.mark.asyncio
async def test_update_single_vector_invalid_name(client: QdrantClient, test_anime_data: List[Dict]):
    """Test updating with invalid vector name."""
    anime_dict = test_anime_data[0]
    anime = AnimeEntry(**anime_dict)

    # Generate dummy vector
    dummy_vector = [0.0] * 1024

    # Try to update non-existent vector
    success = await client.update_single_vector(
        anime_id=anime.id,
        vector_name='invalid_vector',
        vector_data=dummy_vector
    )

    assert success is False, "Invalid vector name should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors(client: QdrantClient, test_anime_data: List[Dict]):
    """Test batch updating multiple vectors."""
    batch_updates = []

    for anime_dict in test_anime_data:
        anime = AnimeEntry(**anime_dict)

        # Generate genre vector
        genre_content = client.embedding_manager.field_mapper._extract_genre_content(anime)
        genre_vector = client.embedding_manager.text_processor.encode_text(genre_content)

        batch_updates.append({
            'anime_id': anime.id,
            'vector_name': 'genre_vector',
            'vector_data': genre_vector
        })

    # Execute batch update
    result = await client.update_batch_vectors(batch_updates)

    assert 'success' in result, "Result should contain success count"
    assert 'failed' in result, "Result should contain failed count"
    assert result['success'] == len(batch_updates), "All updates should succeed"
    assert result['failed'] == 0, "No updates should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_mixed(client: QdrantClient, test_anime_data: List[Dict]):
    """Test batch updating multiple vectors for same anime."""
    anime_dict = test_anime_data[0]
    anime = AnimeEntry(**anime_dict)

    # Generate multiple vectors
    title_content = client.embedding_manager.field_mapper._extract_title_content(anime)
    title_vector = client.embedding_manager.text_processor.encode_text(title_content)

    genre_content = client.embedding_manager.field_mapper._extract_genre_content(anime)
    genre_vector = client.embedding_manager.text_processor.encode_text(genre_content)

    character_content = client.embedding_manager.field_mapper._extract_character_content(anime)
    character_vector = client.embedding_manager.text_processor.encode_text(character_content)

    batch_updates = [
        {
            'anime_id': anime.id,
            'vector_name': 'title_vector',
            'vector_data': title_vector
        },
        {
            'anime_id': anime.id,
            'vector_name': 'genre_vector',
            'vector_data': genre_vector
        },
        {
            'anime_id': anime.id,
            'vector_name': 'character_vector',
            'vector_data': character_vector
        }
    ]

    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 3, "All 3 vector updates should succeed"
    assert result['failed'] == 0, "No updates should fail"


@pytest.mark.asyncio
async def test_update_batch_vectors_empty(client: QdrantClient):
    """Test batch update with empty list."""
    result = await client.update_batch_vectors([])

    assert result['success'] == 0, "Empty batch should have 0 successes"
    assert result['failed'] == 0, "Empty batch should have 0 failures"


@pytest.mark.asyncio
async def test_update_image_vector(client: QdrantClient, test_anime_data: List[Dict]):
    """Test updating image vector."""
    anime_dict = test_anime_data[0]
    anime = AnimeEntry(**anime_dict)

    # Generate image vector
    image_vector = await client.embedding_manager.vision_processor.process_anime_image_vector(anime)

    if image_vector is None:
        pytest.skip("No image data available for this anime")

    # Update image vector
    success = await client.update_single_vector(
        anime_id=anime.id,
        vector_name='image_vector',
        vector_data=image_vector
    )

    assert success is True, "Image vector update should succeed"


@pytest.mark.asyncio
async def test_update_all_text_vectors(client: QdrantClient, test_anime_data: List[Dict]):
    """Test updating all text vectors for one anime."""
    anime_dict = test_anime_data[0]
    anime = AnimeEntry(**anime_dict)

    text_vector_names = [
        'title_vector',
        'character_vector',
        'genre_vector',
        'staff_vector',
        'temporal_vector',
        'streaming_vector',
        'related_vector',
        'franchise_vector',
        'episode_vector'
    ]

    batch_updates = []

    for vector_name in text_vector_names:
        # Extract content based on vector type
        if vector_name == 'title_vector':
            content = client.embedding_manager.field_mapper._extract_title_content(anime)
        elif vector_name == 'character_vector':
            content = client.embedding_manager.field_mapper._extract_character_content(anime)
        elif vector_name == 'genre_vector':
            content = client.embedding_manager.field_mapper._extract_genre_content(anime)
        elif vector_name == 'staff_vector':
            content = client.embedding_manager.field_mapper._extract_staff_content(anime)
        elif vector_name == 'temporal_vector':
            content = client.embedding_manager.field_mapper._extract_temporal_content(anime)
        elif vector_name == 'streaming_vector':
            content = client.embedding_manager.field_mapper._extract_streaming_content(anime)
        elif vector_name == 'related_vector':
            content = client.embedding_manager.field_mapper._extract_related_content(anime)
        elif vector_name == 'franchise_vector':
            content = client.embedding_manager.field_mapper._extract_franchise_content(anime)
        elif vector_name == 'episode_vector':
            content = client.embedding_manager.field_mapper._extract_episode_content(anime)

        # Generate vector
        vector_data = client.embedding_manager.text_processor.encode_text(content)

        batch_updates.append({
            'anime_id': anime.id,
            'vector_name': vector_name,
            'vector_data': vector_data
        })

    # Execute batch update
    result = await client.update_batch_vectors(batch_updates)

    assert result['success'] == 9, "All 9 text vector updates should succeed"
    assert result['failed'] == 0, "No updates should fail"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
