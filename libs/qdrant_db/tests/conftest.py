"""Test configuration for qdrant_db library tests.

Provides fixtures for testing QdrantClient functionality.
QdrantClient is a database client - it works with raw vectors/points, not processors.
"""

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from common.config.settings import get_settings
from qdrant_db import QdrantClient


@pytest.fixture(scope="session")
def settings():
    """Get test settings with test collection name."""
    settings = get_settings()
    # Override to use test collection for libs tests
    settings.qdrant_collection_name = "anime_database_test"
    return settings


@pytest_asyncio.fixture(scope="session")
async def client(settings):
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

        # Initialize our QdrantClient wrapper - NOTE: no embedding_manager parameter
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
