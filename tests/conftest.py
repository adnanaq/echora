"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient
from src.vector.processors.embedding_manager import MultiVectorEmbeddingManager
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor


@pytest.fixture
def mock_redis_cache_miss():
    """
    Ensure any result cache lookup misses by patching the Redis client used by the result cache.

    This pytest fixture patches src.cache_manager.result_cache.get_result_cache_redis_client to return an AsyncMock Redis client whose `get` method always returns `None`, causing cached result lookups to behave as cache misses for the duration of the test.

    Yields:
        AsyncMock: The mocked Redis client, allowing tests to assert on call counts or behavior.
    """
    with patch(
        "src.cache_manager.result_cache.get_result_cache_redis_client"
    ) as mock_get_redis_client:
        mock_redis_client = AsyncMock()
        mock_redis_client.get.return_value = None  # Always return None for get
        mock_get_redis_client.return_value = mock_redis_client
        yield mock_redis_client


@pytest.fixture(scope="session")
def settings():
    """
    Provide application settings configured to use the test Qdrant collection.
    
    Overrides the `qdrant_collection_name` attribute to "anime_database_test" so all tests operate against the dedicated test collection.
    
    Returns:
        settings: Settings instance with `qdrant_collection_name` set to "anime_database_test".
    """
    settings = get_settings()
    # Override to use test collection for ALL tests
    settings.qdrant_collection_name = "anime_database_test"
    return settings


@pytest_asyncio.fixture(scope="session")
async def text_processor(settings):
    """Create TextProcessor for tests."""
    return TextProcessor(settings)


@pytest_asyncio.fixture(scope="session")
async def vision_processor(settings):
    """Create VisionProcessor for tests."""
    return VisionProcessor(settings)


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
    except Exception as e:
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
