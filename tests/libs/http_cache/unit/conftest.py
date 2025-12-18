"""Shared test fixtures and utilities for http_cache unit tests.

This module provides common fixtures and helper classes used across multiple
http_cache unit test files to reduce code duplication and ensure consistency.
"""

from __future__ import annotations

import pytest
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from hishel._core.models import Headers, Request, Response

from http_cache.async_redis_storage import AsyncRedisStorage


# ============================================================================
# Mock Redis Client Fixtures
# ============================================================================


@pytest.fixture
def mock_redis_client() -> AsyncMock:
    """
    Create a mock Async Redis client preconfigured for tests.

    The mock exposes common async Redis methods with sensible default return
    values to simplify unit tests.

    Returns:
        AsyncMock: A mock Redis client with methods like `from_url`, `aclose`,
        `pipeline`, `hset`, `hgetall` (returns `{}`), `sadd`, `smembers`
        (returns `set()`), `srem`, `expire`, `rpush`, `lrange` (returns `[]`),
        `scan` (returns `(0, [])`), and `delete` already set as AsyncMock/MagicMock.
    """
    mock_client = AsyncMock()
    mock_client.from_url = AsyncMock(return_value=mock_client)
    mock_client.aclose = AsyncMock()
    mock_client.pipeline = MagicMock(return_value=AsyncMock())
    mock_client.hset = AsyncMock()
    mock_client.hgetall = AsyncMock(return_value={})
    mock_client.sadd = AsyncMock()
    mock_client.smembers = AsyncMock(return_value=set())
    mock_client.srem = AsyncMock()
    mock_client.expire = AsyncMock()
    mock_client.rpush = AsyncMock()
    mock_client.lrange = AsyncMock(return_value=[])
    mock_client.scan = AsyncMock(return_value=(0, []))
    mock_client.delete = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.setex = AsyncMock()
    return mock_client


@pytest.fixture
def storage_with_mock_client(mock_redis_client: AsyncMock) -> AsyncRedisStorage:
    """
    Create an AsyncRedisStorage configured with a mocked Redis client for tests.

    Parameters:
        mock_redis_client: Mocked AsyncRedis client to be used by the storage.

    Returns:
        AsyncRedisStorage: Storage instance configured for testing
        (default_ttl=3600.0, refresh_ttl_on_access=True, key_prefix="test_cache").
    """
    return AsyncRedisStorage(
        client=mock_redis_client,
        default_ttl=3600.0,
        refresh_ttl_on_access=True,
        key_prefix="test_cache",
    )


# ============================================================================
# Hishel Mock Request/Response Fixtures
# ============================================================================


@pytest.fixture
def mock_request() -> Request:
    """
    Create a mock GET Request targeting a sample resource.

    Returns:
        Request: A Request with method "GET", URL "https://api.example.com/anime/1",
        empty headers, and empty metadata.
    """
    return Request(
        method="GET",
        url="https://api.example.com/anime/1",
        headers=Headers({}),
        metadata={},
    )


@pytest.fixture
def mock_response() -> Response:
    """
    Create a mock HTTP Response containing a three-chunk async body stream.

    Returns:
        Response: A Response with status_code 200, a `Content-Type: application/json`
        header, an async stream that yields `b"chunk1"`, `b"chunk2"`, and `b"chunk3"`,
        and empty metadata.
    """

    async def mock_stream() -> AsyncIterator[bytes]:
        """Generate a three-chunk async stream."""
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    return Response(
        status_code=200,
        headers=Headers({"content-type": "application/json"}),
        stream=mock_stream(),
        metadata={},
    )


# ============================================================================
# Mock Storage Helper Classes
# ============================================================================


class MockAsyncStorage:
    """
    Simple in-memory storage implementation for testing aiohttp adapter.

    Provides get_entries/create_entry methods that simulate Hishel's storage
    interface for testing CachedAiohttpSession without real storage dependencies.
    """

    def __init__(self):
        """
        Initialize the instance and create an empty in-memory storage mapping
        for cache entries.
        """
        self._storage: dict = {}

    async def get_entries(self, key: str):
        """
        Retrieve cached entries for the given cache key.

        Parameters:
            key: Cache key to look up.

        Returns:
            list: The list of cache entries associated with `key`, or `None`
            if no entries exist.
        """
        return self._storage.get(key)

    async def create_entry(self, _request, response, key):
        """
        Create and store a mock cache entry by consuming the provided response
        stream and materializing its body.

        Parameters:
            _request: The original request object (kept for interface compatibility;
                not inspected by this helper).
            response: An object exposing `status_code`, `headers`, and `stream`
                (an async or sync iterable of bytes) whose stream will be consumed
                and materialized.
            key: The storage key under which the created entry will be inserted.

        Returns:
            MagicMock: A mock entry whose `.response` has `status_code`, `headers`,
            and a `.stream` set to a list containing the full body bytes; the entry
            is inserted at the front of the storage list for the given key.
        """
        body_chunks = []
        # Consume the stream
        stream = response.stream
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                body_chunks.append(chunk)
        else:
            # Sync stream
            for chunk in stream:
                body_chunks.append(chunk)

        body = b"".join(body_chunks)

        # Create mock entry
        entry = MagicMock()
        entry.response = MagicMock()
        entry.response.status_code = response.status_code
        entry.response.headers = response.headers
        entry.response.stream = [body]  # Store as list of full body

        # Add to storage
        if key not in self._storage:
            self._storage[key] = []
        self._storage[key].insert(0, entry)
        return entry

    async def close(self):
        """Close storage (no-op for mock)."""
        pass


@pytest.fixture
def mock_storage() -> AsyncMock:
    """
    Provide an AsyncMock-wrapped MockAsyncStorage suitable for tests.

    Returns:
        AsyncMock: An AsyncMock wrapping a MockAsyncStorage instance. The wrapper
        preserves MockAsyncStorage behavior and exposes a replaceable `close`
        coroutine mocked as an AsyncMock.
    """
    storage = MockAsyncStorage()
    mock = AsyncMock(wraps=storage)
    mock.close = AsyncMock()
    return mock


# ============================================================================
# Result Cache Test Utilities
# ============================================================================


@pytest.fixture(autouse=True)
def reset_result_cache_redis_client():
    """
    Automatically reset the _redis_client singleton before each test.

    This ensures a clean state for result_cache tests and prevents test
    pollution from shared module-level state.
    """
    import http_cache.result_cache

    http_cache.result_cache._redis_client = None
    yield
    # Cleanup after test
    http_cache.result_cache._redis_client = None
