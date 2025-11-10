"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient


@pytest.fixture
def mock_redis_cache_miss():
    """Fixture to ensure cached_result always results in a cache miss."""
    with patch(
        "src.cache_manager.result_cache.get_result_cache_redis_client"
    ) as mock_get_redis_client:
        mock_redis_client = AsyncMock()
        mock_redis_client.get.return_value = None  # Always return None for get
        mock_get_redis_client.return_value = mock_redis_client
        yield


@pytest.fixture(scope="session")
def settings():
    """Get test settings with test collection name."""
    settings = get_settings()
    # Override to use test collection for ALL tests
    settings.qdrant_collection_name = "anime_database_test"
    return settings


@pytest_asyncio.fixture
async def client(settings):
    """Create QdrantClient with test collection.

    Collection is automatically created/validated during client initialization.
    """
    try:
        client = QdrantClient(settings=settings)
    except Exception as e:
        pytest.skip(f"Failed to create test collection: {e}")

    yield client

    # Cleanup: Delete test collection after tests for isolation
    try:
        await client.delete_collection()
    except Exception:
        # Ignore cleanup errors to avoid test failures
        pass
