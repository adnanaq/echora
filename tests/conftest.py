"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

import pytest
import pytest_asyncio

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient


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
