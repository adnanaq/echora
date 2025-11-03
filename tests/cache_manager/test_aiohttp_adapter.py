"""
Tests for aiohttp caching adapter.

Tests the CachedAiohttpSession wrapper that adds HTTP caching via Redis.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from src.cache_manager.aiohttp_adapter import (
    _CachedResponse,
    _CachedRequestContextManager,
    CachedAiohttpSession,
)


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
        )

        # Case-insensitive access
        assert response.headers["content-type"] == "application/json"
        assert response.headers["CONTENT-TYPE"] == "application/json"
        assert isinstance(response.headers, CIMultiDictProxy)

    @pytest.mark.asyncio
    async def test_read(self):
        """Test read() method returns body."""
        body = b"test data"
        response = _CachedResponse(200, {}, body, "https://example.com")

        result = await response.read()
        assert result == body

    @pytest.mark.asyncio
    async def test_text(self):
        """Test text() method decodes body as UTF-8."""
        response = _CachedResponse(200, {}, b"Hello World", "https://example.com")

        text = await response.text()
        assert text == "Hello World"

    @pytest.mark.asyncio
    async def test_text_custom_encoding(self):
        """Test text() with custom encoding."""
        response = _CachedResponse(200, {}, "Héllo".encode("latin-1"), "https://example.com")

        text = await response.text(encoding="latin-1")
        assert text == "Héllo"

    @pytest.mark.asyncio
    async def test_json(self):
        """Test json() method parses JSON body."""
        body = b'{"key": "value", "number": 42}'
        response = _CachedResponse(200, {}, body, "https://example.com")

        data = await response.json()
        assert data == {"key": "value", "number": 42}

    def test_release(self):
        """Test release() sets _released flag."""
        response = _CachedResponse(200, {}, b"", "https://example.com")
        assert response._released is False

        response.release()
        assert response._released is True

    def test_raise_for_status_success(self):
        """Test raise_for_status() doesn't raise for 2xx/3xx."""
        response_200 = _CachedResponse(200, {}, b"", "https://example.com")
        response_200.raise_for_status()  # Should not raise

        response_302 = _CachedResponse(302, {}, b"", "https://example.com")
        response_302.raise_for_status()  # Should not raise

    def test_raise_for_status_error(self):
        """Test raise_for_status() raises for 4xx/5xx."""
        response_404 = _CachedResponse(404, {}, b"", "https://example.com")
        with pytest.raises(ValueError, match="HTTP 404"):
            response_404.raise_for_status()

        response_500 = _CachedResponse(500, {}, b"", "https://example.com")
        with pytest.raises(ValueError, match="HTTP 500"):
            response_500.raise_for_status()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test _CachedResponse works as async context manager."""
        response = _CachedResponse(200, {}, b"test", "https://example.com")

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
        mock_response = _CachedResponse(200, {}, b"test", "https://example.com")

        # Create a proper coroutine that returns the response
        async def mock_coro():
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
        mock_response = _CachedResponse(200, {}, b"test", "https://example.com")

        # Create a proper coroutine that returns the response
        async def mock_coro():
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

    @pytest.fixture
    def mock_storage(self):
        """Create mock async storage."""
        storage = AsyncMock()
        storage.get_entries = AsyncMock(return_value=[])
        storage.create_entry = AsyncMock()
        return storage

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
        """Test initialization creates session if not provided."""
        with patch('src.cache_manager.aiohttp_adapter.aiohttp.ClientSession') as mock_client_session:
            mock_session_instance = AsyncMock()
            mock_client_session.return_value = mock_session_instance

            cached_session = CachedAiohttpSession(
                storage=mock_storage,
                timeout=MagicMock(),
            )

            assert cached_session.storage is mock_storage
            mock_client_session.assert_called_once()

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
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        key = cached_session._generate_cache_key("GET", "https://example.com/api", {})

        assert key.startswith("GET:")
        assert len(key) > 4  # Has hash

    def test_generate_cache_key_post_with_json(self, mock_storage):
        """Test cache key generation for POST with JSON body."""
        # Pass a mock session to avoid event loop requirement
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        key1 = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"json": {"key": "value1"}}
        )
        key2 = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"json": {"key": "value2"}}
        )

        assert key1.startswith("POST:")
        assert key2.startswith("POST:")
        assert key1 != key2  # Different bodies = different keys

    def test_generate_cache_key_post_with_data(self, mock_storage):
        """Test cache key generation for POST with form data."""
        # Pass a mock session to avoid event loop requirement
        mock_session = MagicMock()
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        key = cached_session._generate_cache_key(
            "POST", "https://example.com/api", {"data": "form=data"}
        )

        assert key.startswith("POST:")

    @pytest.mark.asyncio
    async def test_close(self, mock_storage):
        """Test close() closes session and storage."""
        mock_session = AsyncMock()
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

        async with CachedAiohttpSession(storage=mock_storage, session=mock_session) as session:
            assert isinstance(session, CachedAiohttpSession)

        # Should close on exit
        mock_session.close.assert_awaited_once()
        mock_storage.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_cache_miss(self, mock_storage):
        """Test _request with cache miss - makes HTTP request."""
        # Setup: No cache entries (cache miss)
        mock_storage.get_entries = AsyncMock(return_value=[])
        mock_storage.create_entry = AsyncMock()

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://example.com/api"
        mock_response.read = AsyncMock(return_value=b'{"data": "test"}')
        mock_response.request_info = MagicMock()
        mock_response.request_info.headers = {}

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)

        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

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
        from hishel import Response, Headers

        async def mock_stream():
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

        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

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

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_response)

        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

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
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # Execute store
        await cached_session._store_response_with_body(
            method="GET",
            url="https://example.com/api",
            response=mock_response,
            cache_key="GET:test123",
            request_kwargs={},
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
            yield b'{"sync": "data"}'

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = Headers({"Content-Type": "text/plain"})
        mock_hishel_response.stream = mock_sync_stream()  # Sync iterator

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

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
            yield b'{"dict": "headers"}'

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = {"Content-Type": "application/json"}  # Plain dict
        mock_hishel_response.stream = mock_stream()

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

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
            yield b'test'

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
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # Execute - should handle sync iterator without error
        await cached_session._store_response_with_body(
            method="POST",
            url="https://example.com",
            response=mock_response,
            cache_key="POST:abc",
            request_kwargs={},
            body=b'test',
        )

        mock_storage.create_entry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_request_cache_hit_headers_exception_fallback(self, mock_storage):
        """Test _request cache hit with Headers that trigger exception (lines 243-247)."""
        from hishel import Headers

        async def mock_stream():
            yield b'{"exception": "test"}'

        # Create Headers with _headers that will cause exception during iteration
        mock_headers = Headers({"Content-Type": "application/json"})
        # Mock _headers to raise exception during iteration
        mock_headers._headers = MagicMock()
        mock_headers._headers.__iter__ = MagicMock(side_effect=ValueError("Test exception"))

        mock_hishel_response = MagicMock()
        mock_hishel_response.status_code = 200
        mock_hishel_response.headers = mock_headers
        mock_hishel_response.stream = mock_stream()

        mock_entry = MagicMock()
        mock_entry.response = mock_hishel_response

        mock_storage.get_entries = AsyncMock(return_value=[mock_entry])

        mock_session = AsyncMock()
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

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

        async def capture_create_entry(request, response, cache_key):
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
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # Execute store
        body_data = b'{"factory": "test"}'
        await cached_session._store_response_with_body(
            method="GET",
            url="https://example.com/api",
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
        assert b''.join(chunks) == body_data

    @pytest.mark.asyncio
    async def test_request_cache_hit_headers_list_extraction(self, mock_storage):
        """Test _request cache hit with Headers._headers as list (lines 243-244)."""
        from hishel import Headers

        async def mock_stream():
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
        cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # Execute request - should extract headers from list structure
        result = await cached_session._request("GET", "https://example.com/api")

        # Verify headers were extracted successfully
        assert result.status == 200
        assert result.from_cache is True
        assert "Content-Type" in result.headers
        assert "Cache-Control" in result.headers
