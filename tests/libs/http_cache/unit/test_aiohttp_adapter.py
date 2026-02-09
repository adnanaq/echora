"""
Tests for aiohttp caching adapter.

Tests the CachedAiohttpSession wrapper that adds HTTP caching via Redis.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hishel import Headers, Request, Response
from http_cache.aiohttp_adapter import (
    CachedAiohttpSession,
    _CachedRequestContextManager,
    _CachedResponse,
)
from multidict import CIMultiDict, CIMultiDictProxy
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
        mock_response = _CachedResponse(
            200, {}, b"test", "https://example.com", method="GET", request_headers={}
        )

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

    @pytest.mark.asyncio
    async def test_init_with_session(self, mock_storage):
        """Test initialization with provided session."""
        mock_session = AsyncMock()

        cached_session = CachedAiohttpSession(
            storage=mock_storage,
            session=mock_session,
            force_cache=True,
            always_revalidate=True,
        )

        assert cached_session.storage is mock_storage
        assert cached_session.session is mock_session
        assert cached_session.force_cache is True
        assert cached_session.always_revalidate is True

    @pytest.mark.asyncio
    async def test_init_creates_session(self, mock_storage):
        """Test session creation during __init__."""
        with patch(
            "http_cache.aiohttp_adapter.aiohttp.ClientSession"
        ) as mock_client_session:
            mock_session_instance = AsyncMock()
            mock_client_session.return_value = mock_session_instance

            cached_session = CachedAiohttpSession(
                storage=mock_storage,
            )

            assert cached_session.storage is mock_storage
            assert cached_session.session is mock_session_instance
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

    @pytest.mark.asyncio
    async def test_close(self, mock_storage):
        """Test close() closes session and storage."""
        mock_session = AsyncMock()
        mock_session.closed = False
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
        mock_session.closed = False

        async with CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        ) as session:
            assert isinstance(session, CachedAiohttpSession)

        mock_session.close.assert_awaited_once()
        mock_storage.close.assert_awaited_once()

        @pytest.mark.asyncio

        async def test_request_cache_miss(self, mock_storage):

            """Test _request with cache miss - makes HTTP request via proxy."""

            # Mock aiohttp response

            mock_response = MagicMock()

            mock_response.status = 200

            mock_response.headers = {"Content-Type": "application/json"}

            mock_response.url = URL("https://example.com/api")

            mock_response.read = AsyncMock(return_value=b'{"data": "test"}')

            mock_response.request_info = MagicMock()

            mock_response.request_info.headers = {"User-Agent": "test"}

            mock_response.close = MagicMock()

    

            mock_session = AsyncMock()

            # aiohttp session.request returns a response object (our adapter uses it directly)

            mock_session.request = AsyncMock(return_value=mock_response)

    

            cached_session = CachedAiohttpSession(

                storage=mock_storage, session=mock_session

            )

    

            # Execute request

            result = await cached_session._request("GET", "https://example.com/api")

    

            # Verify response attributes

            assert isinstance(result, _CachedResponse)

            assert result.status == 200

            assert result.from_cache is False

            assert await result.read() == b'{"data": "test"}'

    

        @pytest.mark.asyncio

        async def test_request_cache_hit(self, mock_storage):

            """Test _request with cache hit."""

            # Setup: Mock cache entry

            async def mock_stream():

                yield b'{"cached": "data"}'

    

            mock_entry_request = Request(method="GET", url="https://example.com/api", headers=Headers({}))

            mock_entry_response = Response(

                status_code=200,

                headers=Headers({"Content-Type": "application/json", "Cache-Control": "max-age=3600"}),

                stream=mock_stream(),

            )

    

            # Store in mock storage

            await mock_storage.create_entry(mock_entry_request, mock_entry_response, "mock_key")

    

            # Mock aiohttp response for potential internal calls (should NOT be called for fresh hit)

            mock_response = MagicMock()

            mock_response.headers = {}

            mock_response.close = MagicMock()

    

            mock_session = AsyncMock()

            mock_session.request = AsyncMock(return_value=mock_response)

            

            cached_session = CachedAiohttpSession(

                storage=mock_storage, session=mock_session

            )

    

            # Execute request

            result = await cached_session._request("GET", "https://example.com/api")

    

            # Verify NO HTTP request was made (hit)

            mock_session.request.assert_not_called()

    

                    # Verify cached response was returned

    

                    assert result.status == 200

    

                    assert result.from_cache is True

    

                    assert await result.read() == b'{"cached": "data"}'

    

            

    

                @pytest.mark.asyncio

    

                async def test_request_with_json_body(self, mock_storage):

    

                    """Test _request with JSON body."""

    

                    mock_response = MagicMock()

    

                    mock_response.status = 200

    

                    mock_response.headers = {}

    

                    mock_response.read = AsyncMock(return_value=b"ok")

    

                    mock_response.request_info = MagicMock()

    

                    mock_response.request_info.headers = {}

    

                    mock_response.close = MagicMock()

    

            

    

                    mock_session = AsyncMock()

    

                    mock_session.request = AsyncMock(return_value=mock_response)

    

            

    

                    cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

    

                    await cached_session._request("POST", "https://example.com", json={"foo": "bar"})

    

            

    

                    # Verify session was called with the body

    

                    args, kwargs = mock_session.request.call_args

    

                    assert kwargs["json"] == {"foo": "bar"}

    

            

    

                @pytest.mark.asyncio

    

                async def test_request_with_data_body(self, mock_storage):

    

                    """Test _request with string data body."""

    

                    mock_response = MagicMock()

    

                    mock_response.status = 200

    

                    mock_response.headers = {}

    

                    mock_response.read = AsyncMock(return_value=b"ok")

    

                    mock_response.request_info = MagicMock()

    

                    mock_response.request_info.headers = {}

    

                    mock_response.close = MagicMock()

    

            

    

                    mock_session = AsyncMock()

    

                    mock_session.request = AsyncMock(return_value=mock_response)

    

            

    

                    cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

    

                    await cached_session._request("POST", "https://example.com", data="form_data")

    

            

    

                    # Verify session was called with the data

    

                    args, kwargs = mock_session.request.call_args

    

                    assert kwargs["data"] == "form_data"

    

            

    

                @pytest.mark.asyncio

    

                async def test_request_with_bytes_data_body(self, mock_storage):

    

                    """Test _request with bytes data body."""

    

                    mock_response = MagicMock()

    

                    mock_response.status = 200

    

                    mock_response.headers = {}

    

                    mock_response.read = AsyncMock(return_value=b"ok")

    

                    mock_response.request_info = MagicMock()

    

                    mock_response.request_info.headers = {}

    

                    mock_response.close = MagicMock()

    

            

    

                    mock_session = AsyncMock()

    

                    mock_session.request = AsyncMock(return_value=mock_response)

    

            

    

                    cached_session = CachedAiohttpSession(storage=mock_storage, session=mock_session)

    

                    await cached_session._request("POST", "https://example.com", data=b"binary_data")

    

            

    

                    # Verify session was called with the data

    

                    args, kwargs = mock_session.request.call_args

    

                    assert kwargs["data"] == b"binary_data"

    

            

    

                @pytest.mark.asyncio

    

                async def test_force_cache_injection(self, mock_storage):

    

                    """Test that force_cache injects Cache-Control header when missing."""

    

                    mock_response = MagicMock()

    

                    mock_response.status = 200

    

                    # NO Cache-Control header

    

                    mock_response.headers = {"Content-Type": "application/json"}

    

                    mock_response.read = AsyncMock(return_value=b"{}")

    

                    mock_response.request_info = MagicMock()

    

                    mock_response.request_info.headers = {}

    

                    mock_response.close = MagicMock()

    

            

    

                    mock_session = AsyncMock()

    

                    mock_session.request = AsyncMock(return_value=mock_response)

    

            

    

                    # Enable force_cache

    

                    cached_session = CachedAiohttpSession(

    

                        storage=mock_storage, session=mock_session, force_cache=True

    

                    )

    

                    

    

                    # We need to capture what is stored in mock_storage

    

                    await cached_session._request("GET", "https://example.com")

    

                    

    

                    # Verify that an entry was created in storage

    

                    assert mock_storage.create_entry.called

    

                    

    

                    # Check the headers of the stored response

    

                    stored_response = mock_storage.create_entry.call_args[0][1]

    

                    assert "Cache-Control" in stored_response.headers

    

                    assert "max-age=86400" in stored_response.headers["Cache-Control"]

    

            

    

                @pytest.mark.asyncio

    

                async def test_always_revalidate_logic(self, mock_storage):

    

                    """Test that always_revalidate adds no-cache to request headers."""

    

                    mock_response = MagicMock()

    

                    mock_response.status = 200

    

                    mock_response.headers = {"Cache-Control": "max-age=3600"}

    

                    mock_response.read = AsyncMock(return_value=b"{}")

    

                    mock_response.request_info = MagicMock()

    

                    mock_response.request_info.headers = {}

    

                    mock_response.close = MagicMock()

    

            

    

                    mock_session = AsyncMock()

    

                    mock_session.request = AsyncMock(return_value=mock_response)

    

            

    

                    # Enable always_revalidate

    

                    cached_session = CachedAiohttpSession(

    

                        storage=mock_storage, session=mock_session, always_revalidate=True

    

                    )

    

                    

    

                    await cached_session._request("GET", "https://example.com")

    

                    

    

                    # Check the headers of the request sent to the proxy (via request_sender)

    

                    # request_sender is called when proxy decides to fetch from origin

    

                    args, kwargs = mock_session.request.call_args

    

                    assert kwargs["headers"]["Cache-Control"] == "no-cache"

    

            

    