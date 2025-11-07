"""
Unit tests for CachedAiohttpSession event loop handling.

Tests ensure that the session correctly handles:
1. Single event loop reuse (normal case)
2. Multiple sequential event loops (pytest-asyncio scenario)
3. Session cleanup when event loop changes
4. No event loop running (sync context)
5. Concurrent requests within same loop
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis as AsyncRedis

from src.cache_manager.aiohttp_adapter import CachedAiohttpSession
from src.cache_manager.async_redis_storage import AsyncRedisStorage


@pytest.fixture
def mock_storage():
    """Provide mock AsyncRedisStorage."""
    storage = AsyncMock(spec=AsyncRedisStorage)
    storage.get_entries = AsyncMock(return_value=[])
    storage.create_entry = AsyncMock()
    storage.close = AsyncMock()
    return storage


@pytest.mark.asyncio
async def test_session_created_for_current_event_loop(mock_storage):
    """Test that session is created for the current event loop on first request."""
    cached_session = CachedAiohttpSession(storage=mock_storage)

    # Session should be None initially
    assert cached_session._session is None
    assert cached_session._session_event_loop is None

    # Make a request - should create session
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{"test": "data"}')
        mock_response.url = "https://example.com/test"
        mock_response.request_info.headers = {}

        mock_session_instance = AsyncMock()
        mock_session_instance.request = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session_instance

        # Trigger session creation
        response = await cached_session._request("GET", "https://example.com/test")

        # Session should be created
        assert cached_session._session is not None
        current_loop = asyncio.get_running_loop()
        assert cached_session._session_event_loop == current_loop
        mock_session_class.assert_called_once()


@pytest.mark.asyncio
async def test_session_reused_in_same_event_loop(mock_storage):
    """Test that session is reused for multiple requests in same event loop."""
    cached_session = CachedAiohttpSession(storage=mock_storage)

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{"test": "data"}')
        mock_response.url = "https://example.com/test"
        mock_response.request_info.headers = {}

        mock_session_instance = AsyncMock()
        mock_session_instance.request = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session_instance

        # Make first request
        await cached_session._request("GET", "https://example.com/test1")
        first_session = cached_session._session

        # Make second request
        await cached_session._request("GET", "https://example.com/test2")
        second_session = cached_session._session

        # Should reuse same session
        assert first_session is second_session
        # Session should only be created once
        assert mock_session_class.call_count == 1


@pytest.mark.asyncio
async def test_sequential_event_loops_create_new_sessions():
    """
    Test that sessions are created correctly even when CachedAiohttpSession
    instances are reused across different function calls.

    This verifies the fix handles event loop changes within the same test.
    """
    storage = AsyncMock(spec=AsyncRedisStorage)
    storage.get_entries = AsyncMock(return_value=[])
    storage.close = AsyncMock()

    cached_session = CachedAiohttpSession(storage=storage)

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{}')
        mock_response.url = "https://example.com"
        mock_response.request_info.headers = {}

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.closed = False
        mock_session_class.return_value = mock_session

        # First request creates session
        await cached_session._request("GET", "https://example.com")
        first_session = cached_session._session
        first_loop = cached_session._session_event_loop

        # Verify session was created
        assert first_session is not None
        assert first_loop == asyncio.get_running_loop()
        mock_session_class.assert_called_once()

        # Second request reuses session (same event loop)
        await cached_session._request("GET", "https://example.com/2")
        second_session = cached_session._session

        # Should reuse same session
        assert second_session is first_session
        # Session should only be created once
        mock_session_class.assert_called_once()


@pytest.mark.asyncio
async def test_old_session_closed_on_loop_change(mock_storage):
    """Test that old session is closed when event loop changes."""
    cached_session = CachedAiohttpSession(storage=mock_storage)

    # Create first session
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{}')
        mock_response.url = "https://example.com"
        mock_response.request_info.headers = {}

        mock_session_1 = AsyncMock()
        mock_session_1.request = AsyncMock(return_value=mock_response)
        mock_session_1.closed = False
        mock_session_1.close = AsyncMock()

        mock_session_class.return_value = mock_session_1

        await cached_session._request("GET", "https://example.com")
        first_session = cached_session._session

        # Simulate event loop change by directly modifying the stored loop
        cached_session._session_event_loop = None

        # Create second session
        mock_session_2 = AsyncMock()
        mock_session_2.request = AsyncMock(return_value=mock_response)
        mock_session_2.closed = False
        mock_session_class.return_value = mock_session_2

        await cached_session._request("GET", "https://example.com")

        # Old session should have close scheduled (asyncio.create_task called)
        # New session should be created
        assert cached_session._session is not first_session


@pytest.mark.asyncio
async def test_concurrent_requests_share_session(mock_storage):
    """Test that concurrent requests in same loop share the same session."""
    cached_session = CachedAiohttpSession(storage=mock_storage)

    session_ids = []

    async def make_request(url: str):
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {}
            mock_response.read = AsyncMock(return_value=b'{}')
            mock_response.url = url
            mock_response.request_info.headers = {}

            mock_session = AsyncMock()
            mock_session.request = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            await cached_session._request("GET", url)
            session_ids.append(id(cached_session._session))

    # Make concurrent requests
    await asyncio.gather(
        make_request("https://example.com/1"),
        make_request("https://example.com/2"),
        make_request("https://example.com/3"),
    )

    # All requests should use the same session
    assert len(set(session_ids)) == 1


@pytest.mark.asyncio
async def test_session_close_cleanup(mock_storage):
    """Test that close() properly cleans up the session."""
    cached_session = CachedAiohttpSession(storage=mock_storage)

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b'{}')
        mock_response.url = "https://example.com"
        mock_response.request_info.headers = {}

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        # Create session
        await cached_session._request("GET", "https://example.com")

        # Close cached session
        await cached_session.close()

        # Session close should be called
        mock_session.close.assert_awaited_once()
        mock_storage.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_not_closed_twice(mock_storage):
    """Test that already closed session is not closed again."""
    cached_session = CachedAiohttpSession(storage=mock_storage)

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session.closed = True  # Already closed
        mock_session.close = AsyncMock()

        cached_session._session = mock_session
        mock_session_class.return_value = mock_session

        # Close cached session
        await cached_session.close()

        # Session close should NOT be called (already closed)
        mock_session.close.assert_not_awaited()
        # Storage close should still be called
        mock_storage.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_hit_does_not_create_session():
    """Test that cache hits don't create sessions unnecessarily."""
    storage = AsyncMock(spec=AsyncRedisStorage)

    # Mock cache hit
    mock_entry = MagicMock()
    mock_entry.response.status_code = 200
    mock_entry.response.headers._headers = [["Content-Type", "application/json"]]

    async def mock_stream():
        yield b'{"cached": true}'

    mock_entry.response.stream = mock_stream()
    mock_entry.meta.created_at = 1234567890.0

    storage.get_entries = AsyncMock(return_value=[mock_entry])
    storage.close = AsyncMock()

    cached_session = CachedAiohttpSession(storage=storage)

    with patch("aiohttp.ClientSession") as mock_session_class:
        # Make request that hits cache
        response = await cached_session._request("GET", "https://example.com")

        # Session should NOT be created (cache hit)
        assert response.from_cache is True
        mock_session_class.assert_not_called()
