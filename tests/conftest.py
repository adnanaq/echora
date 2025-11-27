"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient


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


@pytest_asyncio.fixture
async def client(settings):
    """
    Provide a QdrantClient configured to use the test collection and ensure the collection is deleted after the test.
    
    If the client cannot be created, the test is skipped.
    
    Returns:
        QdrantClient: An instantiated QdrantClient configured for the test collection.
    """
    try:
        client = QdrantClient(settings=settings)
    except Exception as e:
        pytest.skip(f"Failed to create test collection: {e}")

    yield client

    # Cleanup: Delete test collection after tests for isolation
    try:
        await client.delete_collection()
    except Exception as exc:  # noqa: BLE001
        # Log cleanup errors for visibility, but don't fail the test
        logging.getLogger(__name__).warning(
            f"Cleanup failed for test collection '{settings.qdrant_collection_name}': {exc}"
        )