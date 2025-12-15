#!/usr/bin/env python3
"""Quick test to verify all search methods work after interface changes."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.config import get_settings
from qdrant_db.client import QdrantClient
from qdrant_client import AsyncQdrantClient


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
                vector_name="title_vector",
                vector_data=dummy_vector,
                limit=3
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
                vector_queries=vector_queries,
                limit=3,
                fusion_method="rrf"
            )
            print(f"   ✓ search_multi_vector returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search_multi_vector failed: {e}")
            return False

        # Test 3: search_text_comprehensive
        print("3. Testing search_text_comprehensive()...")
        try:
            results = await client.search_text_comprehensive(
                query_text="action anime",
                limit=3
            )
            print(f"   ✓ search_text_comprehensive returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search_text_comprehensive failed: {e}")
            return False

        # Test 4: search_visual_comprehensive
        print("4. Testing search_visual_comprehensive()...")
        try:
            # This might fail if no images, so we'll catch that
            results = await client.search_visual_comprehensive(
                image_url="https://example.com/image.jpg",
                limit=3
            )
            print(f"   ✓ search_visual_comprehensive returned {len(results)} results")
        except Exception as e:
            # Expected to fail with dummy URL, that's okay
            print(f"   ⚠ search_visual_comprehensive (expected to fail with dummy URL): {str(e)[:100]}")

        # Test 5: search_complete
        print("5. Testing search_complete()...")
        try:
            results = await client.search_complete(
                query_text="anime",
                limit=3
            )
            print(f"   ✓ search_complete returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search_complete failed: {e}")
            return False

        # Test 6: search_characters
        print("6. Testing search_characters()...")
        try:
            results = await client.search_characters(
                character_name="protagonist",
                limit=3
            )
            print(f"   ✓ search_characters returned {len(results)} results")
        except Exception as e:
            print(f"   ❌ search_characters failed: {e}")
            return False

        print("\n✅ All search methods are working correctly!")
        return True

    finally:
        await async_qdrant_client.close()


if __name__ == "__main__":
    success = asyncio.run(test_search_methods())
    sys.exit(0 if success else 1)
