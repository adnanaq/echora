"""Shared test fixtures for enrichment unit tests."""

from typing import Generator
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_redis_cache_miss() -> Generator[AsyncMock, None, None]:
    """
    Ensure any result cache lookup misses by patching the Redis client used by the result cache.

    This pytest fixture patches http_cache.result_cache.get_result_cache_redis_client to return
    an AsyncMock Redis client whose `get` method always returns `None`, causing cached result
    lookups to behave as cache misses for the duration of the test.

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
