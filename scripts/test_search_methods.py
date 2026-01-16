#!/usr/bin/env python3
"""Quick test to verify all search methods work after interface changes."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.config import get_settings
from qdrant_client import AsyncQdrantClient
from qdrant_db.client import QdrantClient


async def test_search_methods():
    """Test all search methods to ensure they still work."""
    print("🔍 Testing QdrantClient search methods after interface changes...\n")

    # Initialize settings
    settings = get_settings()

    # Initialize AsyncQdrantClient
    if settings.qdrant_api_key:
        async_qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    else:
        async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        # Initialize Qdrant client
        client = await QdrantClient.create(
            settings=settings,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
        )

        # Check collection exists
        exists = await client.collection_exists()
        print(f"✓ Collection exists: {exists}")

        if not exists:
            print("❌ Collection doesn't exist. Run reindex_anime_database.py first.")
            return False

        # Get stats
        stats = await client.get_stats()
        print(f"✓ Collection stats: {stats.get('points_count', 0)} points\n")

        # Test 1: search_single_vector
        print("1. Testing search_single_vector()...")
        try:
            # Create a dummy vector for title_vector (1024 dimensions)
            dummy_vector = [0.1] * 1024
            results = await client.search_single_vector(
                vector_name="title_vector", vector_data=dummy_vector, limit=3
            )
            print(f"   ✓ search_single_vector returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search_single_vector failed: {e}")
            return False

        # Test 2: search_multi_vector
        print("2. Testing search_multi_vector()...")
        try:
            vector_queries = [
                {"vector_name": "title_vector", "vector_data": [0.1] * 1024},
                {"vector_name": "synopsis_vector", "vector_data": [0.1] * 1024},
            ]
            results = await client.search_multi_vector(
                vector_queries=vector_queries, limit=3, fusion_method="rrf"
            )
            print(f"   ✓ search_multi_vector returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search_multi_vector failed: {e}")
            return False

        # Test 3: search (text-only, replaces search_text_comprehensive)
        print("3. Testing search() with text_embedding only...")
        try:
            # Create dummy text embedding (1024 dimensions for BGE-M3)
            dummy_text_embedding = [0.1] * 1024
            results = await client.search(
                text_embedding=dummy_text_embedding, limit=3
            )
            print(f"   ✓ search(text_embedding=...) returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search(text_embedding=...) failed: {e}")
            return False

        # Test 4: search (image-only, replaces search_visual_comprehensive)
        print("4. Testing search() with image_embedding only...")
        try:
            # Create dummy image embedding (768 dimensions for OpenCLIP ViT-L/14)
            dummy_image_embedding = [0.1] * 768
            results = await client.search(
                image_embedding=dummy_image_embedding, limit=3
            )
            print(f"   ✓ search(image_embedding=...) returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search(image_embedding=...) failed: {e}")
            return False

        # Test 5: search (multimodal, replaces search_complete)
        print("5. Testing search() with both text and image embeddings...")
        try:
            # Create dummy embeddings for both text and image
            dummy_text_embedding = [0.1] * 1024
            dummy_image_embedding = [0.1] * 768
            results = await client.search(
                text_embedding=dummy_text_embedding,
                image_embedding=dummy_image_embedding,
                limit=3,
            )
            print(f"   ✓ search(text+image) returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search(text+image) failed: {e}")
            return False

        # Test 6: search (character entity, replaces search_characters)
        print("6. Testing search() with entity_type='character'...")
        try:
            # Create dummy text embedding for character search
            dummy_text_embedding = [0.1] * 1024
            results = await client.search(
                text_embedding=dummy_text_embedding, entity_type="character", limit=3
            )
            print(f"   ✓ search(entity_type='character') returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search(entity_type='character') failed: {e}")
            return False

        print("\n✅ All search methods are working correctly!")
        return True

    finally:
        await async_qdrant_client.close()


if __name__ == "__main__":
    success = asyncio.run(test_search_methods())
    sys.exit(0 if success else 1)
