"""Integration test configuration.

Inherits settings from root conftest.
Provides fixtures for integration testing with real Qdrant and models.
"""

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from typing import List
from common.models.anime import AnimeEntry

from qdrant_db import QdrantClient
from vector_processing import MultiVectorEmbeddingManager
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader


@pytest_asyncio.fixture(scope="session")
async def text_processor(settings):
    """Create TextProcessor for tests."""
    from vector_processing import TextProcessor
    
    # Create text model using factory
    text_model = EmbeddingModelFactory.create_text_model(settings)
    return TextProcessor(model=text_model, settings=settings)


@pytest_asyncio.fixture(scope="session")
async def vision_processor(settings):
    """Create VisionProcessor for tests."""
    from vector_processing import VisionProcessor
    
    # Create vision model and downloader using factory
    vision_model = EmbeddingModelFactory.create_vision_model(settings)
    downloader = ImageDownloader(settings.model_cache_dir)
    return VisionProcessor(model=vision_model, downloader=downloader, settings=settings)


@pytest_asyncio.fixture(scope="session")
async def embedding_manager(text_processor, vision_processor, settings):
    """Create MultiVectorEmbeddingManager for tests."""
    return MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        settings=settings,
    )


@pytest_asyncio.fixture(scope="session")
async def client(settings, embedding_manager):
    """Create QdrantClient with test collection.

    Collection is automatically created/validated during client initialization.
    Uses session scope so collection persists across all tests.
    """
    async_qdrant_client = None
    try:
        # Initialize AsyncQdrantClient from qdrant-client library
        if settings.qdrant_api_key:
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key
            )
        else:
            async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

        # Initialize our QdrantClient wrapper with injected dependencies
        client = await QdrantClient.create(
            settings=settings,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
        )
        # Attach embedding manager manually as it's not part of init but used in tests
        client.embedding_manager = embedding_manager
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to create test collection: {e}\n")
        import traceback
        traceback.print_exc()
        pytest.skip(f"Failed to create test collection: {e}")

    yield client

    # Cleanup: Delete test collection after tests for isolation
    try:
        await client.delete_collection()
    except Exception:
        # Ignore cleanup errors to avoid test failures
        pass

    # Close AsyncQdrantClient connection to release resources
    try:
        if async_qdrant_client:
            await async_qdrant_client.close()
    except Exception:
        # Ignore close errors to avoid test failures
        pass


@pytest_asyncio.fixture
async def add_test_anime(client, embedding_manager):
    """Helper fixture to add anime entries to test collection.
    
    Converts AnimeEntry objects to PointStruct and adds them to the collection.
    This matches the production pattern from reindex_anime_database.py.
    
    Usage:
        await add_test_anime([anime1, anime2, ...])
    """
    
    async def _add_anime(anime_list: List[AnimeEntry]) -> bool:
        """Convert AnimeEntry to PointStruct and add to collection."""
        # Process anime to generate vectors
        processed_batch = await embedding_manager.process_anime_batch(anime_list)
        
        # Convert to PointStruct
        points = []
        for doc_data in processed_batch:
            if doc_data["metadata"].get("processing_failed"):
                continue
            
            point_id = client._generate_point_id(doc_data["payload"]["id"])
            point = PointStruct(
                id=point_id,
                vector=doc_data["vectors"],
                payload=doc_data["payload"],
            )
            points.append(point)
        
        # Add to collection
        return await client.add_documents(points, batch_size=len(points))
    
    return _add_anime
