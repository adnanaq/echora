"""
Root test configuration for all tests.

Provides isolated test collection to avoid touching production data.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

# Pants runs each test file in its own pytest process against one Redis server.
# Integration tests clear the cache to measure hit/miss behaviour, so a shared
# database means one file wipes the cache another is mid-way through filling —
# and db 0 is the developer's own cache. Give every process its own database
# (1-14, never 0) before any module reads REDIS_URL to build its cache client.
#
# A database is claimed, not derived from the process id: two ids that differ
# by the number of databases would otherwise land on the same one. Claims live
# in db 15, which no test writes to, so a test clearing its own database never
# drops another process's claim. A claim expires on its own if the process dies
# before releasing it.
_REDIS_SERVER = "redis://localhost:6379"
_CLAIMS_DB = 15
_TEST_DBS = range(1, _CLAIMS_DB)
_CLAIM_SECONDS = 2 * 60 * 60


def _claim_test_redis_db() -> int:
    """Claim a Redis database no other test process is using.

    Returns:
        The claimed database number. Without Redis, or with every database
        claimed, one derived from the process id: tests that need Redis skip
        when it is down, and the fallback is no worse than before.
    """
    import atexit

    import redis

    try:
        claims = redis.Redis.from_url(
            f"{_REDIS_SERVER}/{_CLAIMS_DB}", socket_connect_timeout=0.5
        )
        for db in _TEST_DBS:
            key = f"echora:test-redis-db:{db}"
            if claims.set(key, os.getpid(), nx=True, ex=_CLAIM_SECONDS):
                atexit.register(claims.delete, key)
                return db
    except redis.RedisError:
        pass
    return _TEST_DBS[os.getpid() % len(_TEST_DBS)]


TEST_REDIS_DB = _claim_test_redis_db()
TEST_REDIS_URL = f"{_REDIS_SERVER}/{TEST_REDIS_DB}"
os.environ["REDIS_URL"] = TEST_REDIS_URL

if TYPE_CHECKING:
    from common.config.settings import Settings
    from qdrant_client import AsyncQdrantClient
    from qdrant_db import QdrantClient
    from vector_processing import (
        AnimeFieldMapper,
        MultiVectorEmbeddingManager,
        TextProcessor,
        VisionProcessor,
    )


@pytest.fixture(scope="session")
def field_mapper() -> AnimeFieldMapper:
    """Create shared AnimeFieldMapper for tests."""
    from vector_processing import AnimeFieldMapper

    return AnimeFieldMapper()


@pytest.fixture
def mock_redis_cache_miss() -> Generator[AsyncMock, None, None]:
    """
    Ensure any result cache lookup misses by patching the Redis client used by the result cache.

    This pytest fixture patches http_cache.result_cache.get_result_cache_redis_client to return an AsyncMock Redis client whose `get` method always returns `None`, causing cached result lookups to behave as cache misses for the duration of the test.

    Yields:
        AsyncMock: The mocked Redis client, allowing tests to assert on call counts or behavior.
    """
    with patch(
        "http_cache.result_cache.get_result_cache_redis_client"
    ) as mock_get_redis_client:
        mock_redis_client = AsyncMock()
        mock_redis_client.get.return_value = None  # Always return None for get
        mock_get_redis_client.return_value = mock_redis_client
        yield mock_redis_client


@pytest.fixture(scope="session")
def settings() -> Settings:
    """
    Provide application settings configured to use the test Qdrant collection.

    Overrides `qdrant_collection_name` so tests never touch production data.
    The name carries the process id because Pants runs each test file in its own
    pytest process against the same Qdrant server: with one shared name, the
    first process to finish deletes the collection the others are still using.

    Returns:
        settings: Settings instance pointing at this process's test collection.
    """
    import os

    from common.config.settings import get_settings

    settings = get_settings()
    # Override to use test collection for ALL tests
    settings.qdrant.qdrant_collection_name = f"anime_database_test_{os.getpid()}"
    return settings


@pytest_asyncio.fixture(scope="session")
async def text_processor(settings: Settings) -> TextProcessor:
    """Create TextProcessor for tests."""
    from vector_processing import TextProcessor
    from vector_processing.embedding_models.factory import EmbeddingModelFactory

    text_model = EmbeddingModelFactory.create_text_model(settings.embedding)
    return TextProcessor(model=text_model, config=settings.embedding)


@pytest_asyncio.fixture(scope="session")
async def vision_processor(settings: Settings) -> VisionProcessor:
    """Create VisionProcessor for tests."""
    from vector_processing import VisionProcessor
    from vector_processing.embedding_models.factory import EmbeddingModelFactory
    from vector_processing.utils.image_downloader import ImageDownloader

    vision_model = EmbeddingModelFactory.create_vision_model(settings.embedding)
    downloader = ImageDownloader(settings.embedding.model_cache_dir)
    return VisionProcessor(
        model=vision_model,
        downloader=downloader,
        config=settings.embedding,
    )


@pytest_asyncio.fixture(scope="session")
async def embedding_manager(
    text_processor: TextProcessor,
    vision_processor: VisionProcessor,
    field_mapper: AnimeFieldMapper,
) -> MultiVectorEmbeddingManager:
    """Create MultiVectorEmbeddingManager for tests."""
    from vector_processing import MultiVectorEmbeddingManager

    return MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        field_mapper=field_mapper,
    )


@pytest_asyncio.fixture(scope="session")
async def client(
    settings: Settings, embedding_manager: MultiVectorEmbeddingManager
) -> AsyncGenerator[QdrantClient, None]:
    """Create QdrantClient with test collection.

    Collection is automatically created/validated during client initialization.
    Uses session scope so collection persists across all tests.

    Args:
        settings: Application settings fixture
        embedding_manager: Unused parameter, declared to ensure embedding models
                          are loaded before client initialization (fixture dependency ordering)
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_db import QdrantClient

    async_qdrant_client: AsyncQdrantClient | None = None

    try:
        if settings.qdrant.qdrant_api_key:
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant.qdrant_url,
                api_key=settings.qdrant.qdrant_api_key,
            )
        else:
            async_qdrant_client = AsyncQdrantClient(url=settings.qdrant.qdrant_url)

        client = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant.qdrant_url,
            collection_name=settings.qdrant.qdrant_collection_name,
        )
    except Exception as e:
        pytest.skip(f"Failed to create test collection: {e}")

    yield client

    try:
        await client.delete_collection()
    except Exception as e:
        print(f"Warning: failed to delete test collection: {e}")

    try:
        if async_qdrant_client:
            await async_qdrant_client.close()
    except Exception as e:
        print(f"Warning: failed to close AsyncQdrantClient: {e}")
