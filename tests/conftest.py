"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

import pytest
import pytest_asyncio
from qdrant_client import QdrantClient as QdrantSDK

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient
from src.vector.processors.embedding_manager import MultiVectorEmbeddingManager
from src.vector.processors.text_processor import TextProcessor
from src.vector.processors.vision_processor import VisionProcessor


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
        # Initialize Qdrant SDK client
        if settings.qdrant_api_key:
            qdrant_sdk_client = QdrantSDK(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key
            )
        else:
            qdrant_sdk_client = QdrantSDK(url=settings.qdrant_url)

        # Initialize embedding manager and processors
        text_processor = TextProcessor(settings)
        vision_processor = VisionProcessor(settings)
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            settings=settings,
        )

        # Initialize Qdrant client with injected dependencies
        client = QdrantClient(
            settings=settings,
            qdrant_sdk_client=qdrant_sdk_client,
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
