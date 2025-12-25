"""
Tests for aiohttp caching adapter.

Tests the CachedAiohttpSession wrapper that adds HTTP caching via Redis.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from http_cache.aiohttp_adapter import (
    CachedAiohttpSession,
    _CachedRequestContextManager,
    _CachedResponse,
)
from multidict import CIMultiDictProxy
from yarl import URL


class TestCachedResponse:
    """Test _CachedResponse class that mocks aiohttp.ClientResponse for cached data."""

    @pytest.mark.asyncio
    async def test_init_basic(self):
        """Test basic initialization."""
        response = _CachedResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=b'{"test": "data"}',
            url="https://example.com/api",
            method="GET",
            request_headers={},
            from_cache=True,
        )

        assert response.status == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response._body == b'{"test": "data"}'
        assert response.url == URL("https://example.com/api")
        assert response.from_cache is True
        assert response._released is False

    @pytest.mark.asyncio
    async def test_headers_case_insensitive(self):
        """Test headers are case-insensitive via CIMultiDictProxy."""
        response = _CachedResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=b"",
            url="https://example.com",
            method="GET",
            request_headers={},
        )

        # Case-insensitive access
        assert response.headers["content-type"] == "application/json"
        assert response.headers["CONTENT-TYPE"] == "application/json"
        assert isinstance(response.headers, CIMultiDictProxy)

    @pytest.mark.asyncio
    async def test_read(self):
        """Test read() method returns body."""
        body = b"test data"
        response = _CachedResponse(
            200, {}, body, "https://example.com", method="GET", request_headers={}
        )

        result = await response.read()
        assert result == body

    @pytest.mark.asyncio
    async def test_text(self):
        """Test text() method decodes body as UTF-8."""
        response = _CachedResponse(
            200,
            {},
            b"Hello World",
            "https://example.com",
            method="GET",
            request_headers={},
        )

        text = await response.text()
        assert text == "Hello World"

    @pytest.mark.asyncio
    async def test_text_custom_encoding(self):
        """Test text() with custom encoding."""
        response = _CachedResponse(
            200,
            {},
            "Héllo".encode("latin-1"),
            "https://example.com",
            method="GET",
            request_headers={},
        )

        text = await response.text(encoding="latin-1")
        assert text == "Héllo"

    @pytest.mark.asyncio
    async def test_json(self):
        """Test json() method parses JSON body."""
        body = b'{"key": "value", "number": 42}'
        response = _CachedResponse(
            200, {}, body, "https://example.com", method="GET", request_headers={}
        )

        data = await response.json()
        assert data == {"key": "value", "number": 42}

    def test_release(self):
        """Test release() sets _released flag."""
        response = _CachedResponse(
            200, {}, b"", "https://example.com", method="GET", request_headers={}
        )
        assert response._released is False

        response.release()
        assert response._released is True

    def test_raise_for_status_success(self):
        """Test raise_for_status() doesn't raise for 2xx/3xx."""
        response_200 = _CachedResponse(
            200, {}, b"", "https://example.com", method="GET", request_headers={}
        )
        response_200.raise_for_status()  # Should not raise

        response_302 = _CachedResponse(
            302, {}, b"", "https://example.com", method="GET", request_headers={}
        )
        response_302.raise_for_status()  # Should not raise

    def test_raise_for_status_raises_client_response_error(self):
        """Test raise_for_status() raises ClientResponseError for 4xx/5xx."""
        from aiohttp import ClientResponseError

        response_404 = _CachedResponse(
            status=404,
            headers={},
            body=b"",
            url="https://example.com",
            method="GET",
            request_headers={},
        )
        with pytest.raises(ClientResponseError) as excinfo:
            response_404.raise_for_status()
        assert excinfo.value.status == 404

        response_500 = _CachedResponse(
            status=500,
            headers={},
            body=b"",
            url="https://example.com",
            method="GET",
            request_headers={},
        )
        with pytest.raises(ClientResponseError) as excinfo:
            response_500.raise_for_status()
        assert excinfo.value.status == 500

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test _CachedResponse works as async context manager."""
        response = _CachedResponse(
            200, {}, b"test", "https://example.com", method="GET", request_headers={}
        )

        async with response as r:
            assert r is response
            assert r._released is False

        # Should be released after exiting context
        assert response._released is True


class TestCachedRequestContextManager:
    """Test _CachedRequestContextManager for 'async with session.get()' syntax."""

    @pytest.mark.asyncio
    async def test_basic_flow(self):
        """Test basic request context manager flow."""
        mock_response = _CachedResponse(
            200, {}, b"test", "https://example.com", method="GET", request_headers={}
        )

        # Create a proper coroutine that returns the response
        async def mock_coro():
            """
            Produce the predefined `mock_response` when awaited.

            Returns:
                mock_response: The mocked response object returned by the coroutine.
            """
            return mock_response

        mock_session = MagicMock()

        cm = _CachedRequestContextManager(
            mock_coro(), mock_session, "GET", "https://example.com", {}
        )

        async with cm as response:
            assert response is mock_response
            assert response.status == 200

    @pytest.mark.asyncio
    async def test_response_released_on_exit(self):
        """Test response is released when exiting context."""
        mock_response = _CachedResponse(
            200, {}, b"test", "https://example.com", method="GET", request_headers={}
        )

        # Create a proper coroutine that returns the response
        async def mock_coro():
            """
            Produce the predefined `mock_response` when awaited.

            Returns:
                mock_response: The mocked response object returned by the coroutine.
            """
            return mock_response

        mock_session = MagicMock()

        cm = _CachedRequestContextManager(
            mock_coro(), mock_session, "POST", "https://example.com", {}
        )

        assert mock_response._released is False
        async with cm as response:
            assert response._released is False
        assert mock_response._released is True


class TestCachedAiohttpSession:
    """Test CachedAiohttpSession wrapper."""

    @pytest.mark.asyncio
    async def test_init_with_session(self, mock_storage):
        """Test initialization with provided session."""
        mock_session = AsyncMock()

        cached_session = CachedAiohttpSession(
            storage=mock_storage,
            session=mock_session,
        )

        assert cached_session.storage is mock_storage
        assert cached_session.session is mock_session

    @pytest.mark.asyncio
    async def test_init_creates_session(self, mock_storage):
        """Test lazy session creation on first request."""
        with patch(
            "http_cache.aiohttp_adapter.aiohttp.ClientSession"
        ) as mock_client_session:
            mock_session_instance = AsyncMock()
            mock_client_session.return_value = mock_session_instance

            cached_session = CachedAiohttpSession(
                storage=mock_storage,
                timeout=MagicMock(),
            )

            # Session IS created during __init__ (not lazy)
            assert cached_session.storage is mock_storage
            assert cached_session.session is mock_session_instance
            mock_client_session.assert_called_once()

            # Session should be created lazily on first request
            # Setup mock for cache miss scenario
            mock_storage.get_entries = AsyncMock(return_value=[])
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {}
            mock_response.url = "https://example.com"
            mock_response.read = AsyncMock(return_value=b"test")
            mock_response.request_info = MagicMock()
            mock_response.request_info.headers = {}
            mock_session_instance.request = AsyncMock(return_value=mock_response)

            # Make a request to verify session works
            await cached_session._request("GET", "https://example.com")

            # Session should still be the same instance (not recreated)
            assert cached_session.session is mock_session_instance
            mock_client_session.assert_called_once()  # Still only called once during __init__

    @pytest.mark.asyncio
    async def test_get_returns_context_manager(self, mock_storage):
        """Test get() returns context manager."""
        cached_session = CachedAiohttpSession(storage=mock_storage)

        cm = cached_session.get("https://example.com/api")

        assert isinstance(cm, _CachedRequestContextManager)
        assert cm._method == "GET"
        assert cm._url == "https://example.com/api"

    @pytest.mark.asyncio
    async def test_post_returns_context_manager(self, mock_storage):
        """Test post() returns context manager."""
        cached_session = CachedAiohttpSession(storage=mock_storage)

        cm = cached_session.post("https://example.com/api", json={"key": "value"})

        assert isinstance(cm, _CachedRequestContextManager)
        assert cm._method == "POST"
        assert cm._url == "https://example.com/api"

    def test_generate_cache_key_get(self, mock_storage):
        """Test cache key generation for GET request."""
        # Pass a mock session to avoid event loop requirement
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        key = cached_session._generate_cache_key("GET", "https://example.com/api", {})

        assert key.startswith("GET:")
        assert len(key) > 4  # Has hash

    def test_generate_cache_key_post_with_json(self, mock_storage):
        """Test cache key generation for POST with JSON body."""
        # Pass a mock session to avoid event loop requirement
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        key1 = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"json": {"key": "value1"}}
        )
        key2 = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"json": {"key": "value2"}}
        )

        assert key1.startswith("POST:")
        assert key2.startswith("POST:")
        assert key1 != key2  # Different bodies = different keys

    def test_generate_cache_key_post_with_form_data_different_bodies(
        self, mock_storage
    ):
        """Test cache key generation for POST with form data (different bodies)."""
        # Pass a mock session to avoid event loop requirement
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        key = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"data": "form=data"}
        )

        assert key.startswith("POST:")

    def test_generate_cache_key_post_with_list_data(self, mock_storage):
        """Test cache key generation for POST with list data."""
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        key = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"data": ["item1", "item2"]}
        )

        assert key.startswith("POST:")
        assert len(key) > 4

    def test_generate_cache_key_post_with_tuple_data(self, mock_storage):
        """Test cache key generation for POST with tuple data."""
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        key = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"data": ("item1", "item2")}
        )

        assert key.startswith("POST:")
        assert len(key) > 4

    @pytest.mark.asyncio
    async def test_close(self, mock_storage):
        """Test close() closes session and storage."""
        mock_session = AsyncMock()
        mock_session.closed = False  # Add closed property
        cached_session = CachedAiohttpSession(
            storage=mock_storage,
            session=mock_session,
        )

        await cached_session.close()

        mock_session.close.assert_awaited_once()
        mock_storage.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_storage):
        """Test CachedAiohttpSession as async context manager."""
        mock_session = AsyncMock()
        mock_session.closed = False  # Add closed property

        async with CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        ) as session:
            assert isinstance(session, CachedAiohttpSession)

        mock_session.close.assert_awaited_once()
        mock_storage.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_cache_miss(self, mock_storage):
        """Test _request with cache miss - makes HTTP request."""
        # Setup: No cache entries (cache miss)
        mock_storage.get_entries = AsyncMock(return_value=[])
        mock_storage.create_entry = AsyncMock()

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://example.com/api"
        mock_response.read = AsyncMock(return_value=b'{"data": "test"}')
        mock_response.request_info = MagicMock()
        mock_response.request_info.headers = {}

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)

        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Mock _get_or_create_session to return our mock session
        cached_session._get_or_create_session = MagicMock(return_value=mock_session)

        # Execute request
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify HTTP request was made
        mock_session.request.assert_awaited_once_with("GET", "https://example.com/api")

        # Verify response attributes
        assert isinstance(result, _CachedResponse)
        assert result.status == 200
        assert result.from_cache is False
        assert await result.read() == b'{"data": "test"}'

    @pytest.mark.asyncio
    async def test_request_cache_hit(self, mock_storage):
        """Test _request with cache hit - returns cached response."""
        # Setup: Mock cache entry (cache hit)
        from hishel import Headers

        async def mock_stream():
            """
            Yield a single bytes chunk containing JSON-formatted cached data.

            Yields:
                bytes: A single chunk b'{"cached": "data"}' representing the cached response body.
            """
            yield b'{"cached": "data"}'

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = Headers({"Content-Type": "application/json"})
        mock_hishel_response.stream = mock_stream()

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        # Mock session (should NOT be called for cache hit)
        mock_session = AsyncMock()
        mock_session.request = AsyncMock()

        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify NO HTTP request was made (cache hit)
        mock_session.request.assert_not_awaited()

        # Verify cached response was returned
        assert isinstance(result, _CachedResponse)
        assert result.status == 200
        assert result.from_cache is True
        assert await result.read() == b'{"cached": "data"}'

    @pytest.mark.asyncio
    async def test_request_error_not_cached(self, mock_storage):
        """Test _request with error response - not cached."""
        # Setup: No cache entries
        mock_storage.get_entries = AsyncMock(return_value=[])
        mock_storage.create_entry = AsyncMock()

        # Mock 404 error response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://example.com/api"
        mock_response.read = AsyncMock(return_value=b'{"error": "not found"}')

        mock_response.request_info = MagicMock()
        mock_response.request_info.headers = {}

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)

        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify response returned but NOT cached (status >= 400)
        assert result.status == 404
        assert result.from_cache is False
        mock_storage.create_entry.assert_not_awaited()  # Should NOT cache errors

    @pytest.mark.asyncio
    async def test_store_response_with_body(self, mock_storage):
        """Test _store_response_with_body stores response in cache."""

        # Mock storage.create_entry to return an Entry with consumable stream
        async def mock_stream():
            """
            Asynchronous generator that yields a single JSON-encoded byte chunk.

            Returns:
                An asynchronous iterator that yields one `bytes` object containing a JSON document (b'{"data": "test"}').
            """
            yield b'{"data": "test"}'

        mock_hishel_response = MagicMock()
        mock_hishel_response.stream = mock_stream()

        # Create a proper mock Entry (don't instantiate real Entry)
        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response
        mock_storage.create_entry = AsyncMock(return_value=mock_entry)

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://example.com/api"
        mock_response.request_info = MagicMock()
        mock_response.request_info.headers = {"User-Agent": "test"}

        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute store
        await cached_session._store_response_with_body(
            method="GET",
            response=mock_response,
            cache_key="GET:test123",
            request_kwargs={"metadata": {"test": "value"}},
            body=b'{"data": "test"}',
        )

        # Verify create_entry was called
        mock_storage.create_entry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_cache_hit_sync_iterator(self, mock_storage):
        """Test _request with cache hit using sync iterator (fallback path)."""
        from hishel import Headers

        # Mock sync iterator (not async)
        def mock_sync_stream():
            """
            Yield a single bytes chunk representing a small JSON payload.

            Yields:
                bytes: A single bytes object containing the JSON b'{"sync": "data"}'.
            """
            yield b'{"sync": "data"}'

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = Headers({"Content-Type": "text/plain"})
        mock_hishel_response.stream = mock_sync_stream()  # Sync iterator

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify cached response with sync iterator handling
        assert isinstance(result, _CachedResponse)
        assert result.status == 200
        assert result.from_cache is True
        assert await result.read() == b'{"sync": "data"}'

    @pytest.mark.asyncio
    async def test_request_cache_hit_dict_headers(self, mock_storage):
        """Test _request cache hit with dict headers (fallback path)."""

        # Mock cache entry with dict headers (not Headers object)
        async def mock_stream():
            """
            Async generator that yields a single JSON-encoded bytes chunk representing a simple header-like object.

            Returns:
                An async iterator that yields one `bytes` value: the JSON-encoded representation of {"dict": "headers"}.
            """
            yield b'{"dict": "headers"}'

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = {
            "Content-Type": "application/json"
        }  # Plain dict
        mock_hishel_response.stream = mock_stream()

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify dict headers fallback works
        assert result.status == 200
        assert result.from_cache is True
        assert await result.read() == b'{"dict": "headers"}'

    @pytest.mark.asyncio
    async def test_store_response_with_body_sync_stream(self, mock_storage):
        """Test _store_response_with_body with sync iterator (fallback path)."""

        # Mock sync iterator in Entry response
        def mock_sync_stream():
            """
            Yield a single bytes chunk "test" to simulate a synchronous response body stream.

            Returns:
                iterator (Iterator[bytes]): An iterator that yields a single `bytes` object `b"test"`.
            """
            yield b"test"

        mock_hishel_response = MagicMock()
        mock_hishel_response.stream = mock_sync_stream()  # Sync iterator

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response
        mock_storage.create_entry = AsyncMock(return_value=mock_entry)

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.url = "https://example.com"
        mock_response.request_info = MagicMock()
        mock_response.request_info.headers = {}

        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute - should handle sync iterator without error
        await cached_session._store_response_with_body(
            method="POST",
            response=mock_response,
            cache_key="POST:abc",
            request_kwargs={},
            body=b"test",
        )

        mock_storage.create_entry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_cache_hit_headers_exception_fallback(self, mock_storage):
        """Test _request cache hit with Headers that trigger exception (lines 243-247)."""
        from hishel import Headers

        async def mock_stream():
            """
            Yield a single bytes payload representing a JSON-encoded exception object.

            Yields:
                bytes: The JSON-encoded byte string b'{"exception": "test"}'.
            """
            yield b'{"exception": "test"}'

        # Create Headers with _headers that will cause exception during iteration
        mock_headers = Headers({"Content-Type": "application/json"})
        # Mock _headers to raise exception during iteration
        mock_headers._headers = MagicMock()
        mock_headers._headers.__iter__ = MagicMock(
            side_effect=ValueError("Test exception")
        )

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = mock_headers
        mock_hishel_response.stream = mock_stream()

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request - should trigger exception fallback
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify fallback handled exception and returned response
        assert result.status == 200
        assert result.from_cache is True

    @pytest.mark.asyncio
    async def test_store_response_factory_yield(self, mock_storage):
        """Test _store_response_with_body exercises body_stream_factory yield (line 346)."""
        # This test ensures the async body_stream() function inside body_stream_factory
        # is actually executed and yields the body data

        # Track what was passed to create_entry
        captured_request = None
        captured_response = None

        async def capture_create_entry(request, response, _cache_key):
            """
            Capture the provided request and response objects and return a mock cache entry that omits a response stream.

            Parameters:
                request: The HTTP request object passed to the cache create routine; stored to `captured_request`.
                response: The HTTP response object passed to the cache create routine; stored to `captured_response`.
                _cache_key: The cache key associated with this create operation (not used).

            Returns:
                mock_entry: A MagicMock instance representing a cache entry whose `response` attribute is set to `None` to indicate no body stream should be consumed.
            """
            nonlocal captured_request, captured_response
            captured_request = request
            captured_response = response

            # Return a mock entry without response stream (to skip consumption)
            mock_entry = MagicMock()
            mock_entry.response = None
            return mock_entry

        mock_storage.create_entry = AsyncMock(side_effect=capture_create_entry)

        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://example.com/api"
        mock_response.request_info = MagicMock()
        mock_response.request_info.headers = {}

        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute store
        body_data = b'{"factory": "test"}'
        await cached_session._store_response_with_body(
            method="GET",
            response=mock_response,
            cache_key="GET:test",
            request_kwargs={},
            body=body_data,
        )

        # Verify create_entry was called
        assert captured_response is not None

        # Now consume the stream factory to trigger line 346 (yield body)
        stream = captured_response.stream
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        # Verify the factory yielded the body data
        assert b"".join(chunks) == body_data

    @pytest.mark.asyncio
    async def test_request_cache_hit_headers_list_extraction(self, mock_storage):
        """Test _request cache hit with Headers._headers as list (lines 243-244)."""
        from hishel import Headers

        async def mock_stream():
            """
            Yield a single bytes chunk containing a JSON-like payload.

            Returns:
                bytes: A bytes object containing the JSON payload b'{"headers": "list"}'.
            """
            yield b'{"headers": "list"}'

        # Create Headers and manually set _headers to a list structure
        mock_headers = Headers({"Content-Type": "application/json"})
        # Simulate Hishel's internal list-based structure
        mock_headers._headers = [
            ["Content-Type", "application/json"],
            ["Content-Length", "123"],
            ("Cache-Control", "max-age=3600"),  # Also test tuple format
        ]

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = mock_headers
        mock_hishel_response.stream = mock_stream()

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request - should extract headers from list structure
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify headers were extracted successfully
        assert result.status == 200
        assert result.from_cache is True
        assert "Content-Type" in result.headers
        assert "Cache-Control" in result.headers

    @pytest.mark.asyncio
    async def test_get_requests_with_different_params_should_not_collide(
        self, mock_storage
    ):
        """
        Verify that GET requests with different query parameters result in
        different cache keys and do not cause a collision.
        """
        # Mock aiohttp session
        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Mock _get_or_create_session to return our mock session
        cached_session._get_or_create_session = MagicMock(return_value=mock_session)

        # --- First Request ---
        url = "http://test.com/api"
        params1 = {"id": 1}
        mock_response1 = AsyncMock()
        mock_response1.status = 200
        mock_response1.headers = {"Content-Type": "application/json"}
        mock_response1.url = f"{url}?id=1"
        mock_response1.read = AsyncMock(return_value=b'{"data": "one"}')
        mock_response1.request_info = MagicMock()
        mock_response1.request_info.headers = {}
        mock_session.request.return_value = mock_response1

        async with cached_session.get(url, params=params1) as resp:
            text = await resp.text()
            assert resp.status == 200
            assert text == '{"data": "one"}'
            assert not resp.from_cache

        mock_session.request.assert_called_once_with("GET", url, params=params1)
        mock_storage.create_entry.assert_awaited_once()

        # --- Second Request (with different params) ---
        params2 = {"id": 2}
        mock_response2 = AsyncMock()
        mock_response2.status = 200
        mock_response2.headers = {"Content-Type": "application/json"}
        mock_response2.url = f"{url}?id=2"
        mock_response2.read = AsyncMock(return_value=b'{"data": "two"}')
        mock_response2.request_info = MagicMock()
        mock_response2.request_info.headers = {}
        mock_session.request.reset_mock()
        mock_session.request.return_value = mock_response2

        async with cached_session.get(url, params=params2) as resp:
            text = await resp.text()
            assert resp.status == 200
            assert text == '{"data": "two"}'
            assert not resp.from_cache

        # Verify that a new real request was made
        mock_session.request.assert_called_once_with("GET", url, params=params2)

    @pytest.mark.asyncio
    async def test_post_requests_with_different_form_data_should_not_collide(
        self, mock_storage
    ):
        """
        Verify that POST requests with different dictionary-based form data
        result in different cache keys and do not cause a collision.
        """
        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Mock _get_or_create_session to return our mock session
        cached_session._get_or_create_session = MagicMock(return_value=mock_session)

        # --- First Request ---
        url = "http://test.com/submit"
        data1 = {"field": "value1"}
        mock_response1 = AsyncMock()
        mock_response1.status = 200
        mock_response1.headers = {}
        mock_response1.url = url
        mock_response1.read = AsyncMock(return_value=b"response1")
        mock_response1.request_info = MagicMock()
        mock_response1.request_info.headers = {}
        mock_session.request.return_value = mock_response1

        async with cached_session.post(url, data=data1) as resp:
            text = await resp.text()
            assert resp.status == 200
            assert text == "response1"
            assert not resp.from_cache

        mock_session.request.assert_called_once_with("POST", url, data=data1)
        mock_storage.create_entry.assert_awaited_once()

        # --- Second Request ---
        data2 = {"field": "value2"}
        mock_response2 = AsyncMock()
        mock_response2.status = 200
        mock_response2.headers = {}
        mock_response2.url = url
        mock_response2.read = AsyncMock(return_value=b"response2")
        mock_response2.request_info = MagicMock()
        mock_response2.request_info.headers = {}
        mock_session.request.reset_mock()
        mock_session.request.return_value = mock_response2

        async with cached_session.post(url, data=data2) as resp:
            text = await resp.text()
            assert resp.status == 200
            assert text == "response2"
            assert not resp.from_cache

        mock_session.request.assert_called_once_with("POST", url, data=data2)

    def test_generate_cache_key_post_with_bytes_data(self, mock_storage):
        """Test cache key generation for POST with bytes data."""
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Test with bytes data (exercises serialize_payload bytes path)
        key1 = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"data": b"raw bytes data"}
        )

        # Test with bytearray data
        key2 = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"data": bytearray(b"bytearray data")}
        )

        assert key1.startswith("POST:")
        assert key2.startswith("POST:")
        assert key1 != key2  # Different bytes = different keys

    @pytest.mark.asyncio
    async def test_cache_hit_returns_newest_entry_not_first_from_smembers(
        self, mock_storage
    ):
        """
        Test that cache hits return the NEWEST entry by created_at timestamp,
        not just entries[0] which may be stale due to Redis SMEMBERS unordered results.

        This test addresses PR #20 discussion r2496155470:
        "AsyncRedisStorage.get_entries iterates over a Redis set (SMEMBERS),
        whose order is undefined. On cache hits you can therefore hand back
        an arbitrarily old response even when a newer entry exists for the same key."
        """
        from hishel import Headers

        # Create three mock entries with different timestamps
        # Simulate Redis returning them in "wrong" order (oldest first)
        async def create_mock_stream(data: bytes):
            """
            Produce an async iterator that yields the provided bytes exactly once.

            Parameters:
                data (bytes): Byte sequence to be yielded by the async iterator.

            Returns:
                An async iterator that yields the provided `data` a single time.
            """
            yield data

        # OLD entry (created_at=1.0)
        mock_old_response = MagicMock()
        mock_old_response.status_code = 200
        mock_old_response.headers = Headers({"Content-Type": "application/json"})
        mock_old_response.stream = create_mock_stream(b'{"data": "old"}')

        mock_old_entry = MagicMock()
        mock_old_entry.response = mock_old_response
        mock_old_entry.meta = MagicMock()
        mock_old_entry.meta.created_at = 1.0

        # MIDDLE entry (created_at=2.0)
        mock_middle_response = MagicMock()
        mock_middle_response.status_code = 200
        mock_middle_response.headers = Headers({"Content-Type": "application/json"})
        mock_middle_response.stream = create_mock_stream(b'{"data": "middle"}')

        mock_middle_entry = MagicMock()
        mock_middle_entry.response = mock_middle_response
        mock_middle_entry.meta = MagicMock()
        mock_middle_entry.meta.created_at = 2.0

        # NEW entry (created_at=3.0)
        mock_new_response = MagicMock()
        mock_new_response.status_code = 200
        mock_new_response.headers = Headers({"Content-Type": "application/json"})
        mock_new_response.stream = create_mock_stream(b'{"data": "new"}')

        mock_new_entry = MagicMock()
        mock_new_entry.response = mock_new_response
        mock_new_entry.meta = MagicMock()
        mock_new_entry.meta.created_at = 3.0

        # Simulate Redis SMEMBERS returning entries in "bad" order (old first)
        mock_storage.get_entries = AsyncMock(
            return_value=[mock_old_entry, mock_middle_entry, mock_new_entry]
        )

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        )

        # Execute request - should get NEWEST entry (created_at=3.0)
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify NO HTTP request was made (cache hit)
        mock_session.request.assert_not_awaited()

        # Verify the NEWEST entry was returned (not entries[0] which is old)
        assert isinstance(result, _CachedResponse)
        assert result.status == 200
        assert result.from_cache is True

        # CRITICAL ASSERTION: Body should be from NEWEST entry
        body = await result.read()
        assert body == b'{"data": "new"}', (
            f"Expected newest entry (created_at=3.0) but got: {body}. "
            "This means entries[0] was used instead of max(entries, key=created_at)"
        )
