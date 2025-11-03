"""
aiohttp caching adapter using Hishel's AsyncRedisStorage.

Provides a drop-in replacement for aiohttp.ClientSession that adds
HTTP caching via Redis backend.

**Note**: This is a simplified implementation that directly uses
AsyncRedisStorage. For async helpers (AniList, Kitsu, AniDB), we use
the cached session which wraps the original aiohttp.ClientSession.
"""

import hashlib
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Optional, Type

import aiohttp
from hishel._core._headers import Headers
from hishel._core._storages._async_base import AsyncBaseStorage
from hishel._core.models import Request, Response
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


from aiohttp import ClientResponseError, RequestInfo


class _CachedResponse:
    """
    Mock aiohttp.ClientResponse for cached data.

    Provides minimal interface needed by enrichment helpers.
    """

    def __init__(
        self,
        status: int,
        headers: Dict[str, str],
        body: bytes,
        url: str,
        method: str,
        request_headers: Dict[str, str],
        from_cache: bool = False,
    ) -> None:
        """
        Initialize cached response.

        Args:
            status: HTTP status code
            headers: Response headers
            body: Response body bytes
            url: Request URL
            method: Request HTTP method
            request_headers: Request headers
            from_cache: Whether this response was served from cache
        """
        self.status = status
        # Create CIMultiDict for case-insensitive header access
        headers_multidict = CIMultiDict(headers)
        self.headers = CIMultiDictProxy(headers_multidict)
        self._body = body
        self.url = URL(url)
        self.method = method
        self.request_headers = CIMultiDictProxy(CIMultiDict(request_headers))
        self._released = False
        self.from_cache = from_cache

    async def read(self) -> bytes:
        """Read response body."""
        return self._body

    async def text(self, encoding: str = "utf-8") -> str:
        """Read response as text."""
        return self._body.decode(encoding)

    async def json(self, **kwargs: Any) -> Any:
        """Read response as JSON."""
        import json

        return json.loads(self._body.decode("utf-8"))

    def release(self) -> None:
        """Release response (no-op for cached responses)."""
        self._released = True

    def raise_for_status(self) -> None:
        """Raise exception for HTTP error status codes."""
        if 400 <= self.status < 600:
            request_info = RequestInfo(
                url=self.url,
                method=self.method,
                headers=self.request_headers,
                real_url=self.url,
            )
            raise ClientResponseError(
                request_info=request_info,
                history=(),
                status=self.status,
                message=f"HTTP {self.status} error",
                headers=self.headers,
            )

    async def __aenter__(self) -> "_CachedResponse":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        self.release()


class _CachedRequestContextManager:
    """
    Async context manager for cached HTTP requests.

    This wrapper allows using `async with session.post()` syntax
    by implementing the async context manager protocol.
    """

    def __init__(
        self,
        coro: Any,  # Coroutine that returns aiohttp.ClientResponse
        session: "CachedAiohttpSession",
        method: str,
        url: str,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize context manager.

        Args:
            coro: Coroutine that makes the actual request
            session: Parent cached session
            method: HTTP method
            url: Request URL
            kwargs: Request arguments
        """
        self._coro = coro
        self._session = session
        self._method = method
        self._url = url
        self._kwargs = kwargs
        self._response: Optional[aiohttp.ClientResponse] = None

    async def __aenter__(self) -> aiohttp.ClientResponse:
        """Enter async context - execute request and return response."""
        self._response = await self._coro
        return self._response

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context - close response if needed."""
        if self._response is not None:
            self._response.release()


class CachedAiohttpSession:
    """
    Wrapper around aiohttp.ClientSession that adds HTTP caching.

    Uses Hishel's AsyncRedisStorage for cache backend with service-specific TTLs.
    """

    def __init__(
        self,
        storage: AsyncBaseStorage,
        session: Optional[aiohttp.ClientSession] = None,
        **session_kwargs: Any,
    ) -> None:
        """
        Initialize cached aiohttp session.

        Args:
            storage: Async storage backend (AsyncRedisStorage)
            session: Existing aiohttp session (optional, created if not provided)
            **session_kwargs: Additional arguments for aiohttp.ClientSession
        """
        self.storage = storage
        self.session = session or aiohttp.ClientSession(**session_kwargs)

    def get(self, url: str, **kwargs: Any) -> _CachedRequestContextManager:
        """
        Perform cached GET request.

        Args:
            url: Request URL
            **kwargs: Additional aiohttp request arguments

        Returns:
            Async context manager for the request
        """
        coro = self._request("GET", url, **kwargs)
        return _CachedRequestContextManager(coro, self, "GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> _CachedRequestContextManager:
        """
        Perform cached POST request.

        Args:
            url: Request URL
            **kwargs: Additional aiohttp request arguments

        Returns:
            Async context manager for the request
        """
        coro = self._request("POST", url, **kwargs)
        return _CachedRequestContextManager(coro, self, "POST", url, kwargs)

    async def _request(
        self, method: str, url: str, **kwargs: Any
    ) -> Any:  # Returns aiohttp.ClientResponse or _CachedResponse
        """
        Internal cached request handler.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional aiohttp request arguments

        Returns:
            aiohttp.ClientResponse or _CachedResponse
        """
        # Generate cache key
        cache_key = self._generate_cache_key(method, url, kwargs)

        # Check cache
        entries = await self.storage.get_entries(cache_key)
        if entries:
            # Cache hit - return cached response WITHOUT making HTTP request
            entry = entries[0]  # Get most recent entry
            if entry.response:
                # Read all chunks from cached stream
                body_chunks: list[bytes] = []
                if entry.response.stream:
                    # Handle both async and sync iterators
                    stream = entry.response.stream
                    if hasattr(stream, "__aiter__"):
                        # AsyncIterator
                        async for chunk in stream:
                            body_chunks.append(chunk)
                    else:
                        # Iterator - convert to async
                        for chunk in stream:
                            body_chunks.append(chunk)
                body = b"".join(body_chunks)

                # Extract headers - convert to simple dict
                if isinstance(entry.response.headers, Headers):
                    # Headers._headers is list-based with nested structure
                    # Convert to dict (taking last value for duplicate keys)
                    headers_dict = {}
                    try:
                        # Try direct iteration first
                        for item in entry.response.headers._headers:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                key, value = item[0], item[1]
                                headers_dict[key] = value
                    except (ValueError, TypeError):
                        # Fallback: headers might be already a dict
                        headers_dict = dict(entry.response.headers._headers)
                else:
                    headers_dict = entry.response.headers

                # Return cached response
                return _CachedResponse(
                    status=entry.response.status_code,
                    headers=headers_dict,
                    body=body,
                    url=url,
                    method=method,
                    request_headers=kwargs.get("headers", {}),
                    from_cache=True,
                )

        # Cache miss - make actual HTTP request
        response = await self.session.request(method, url, **kwargs)

        # Read response body to cache it
        body = await response.read()

        # Only cache successful responses (2xx and 3xx)
        # NEVER cache error responses (4xx, 5xx) as they are temporary
        if response.status < 400:
            await self._store_response_with_body(
                method, url, response, cache_key, kwargs, body
            )

        # Return cached response wrapper (allows multiple reads)
        return _CachedResponse(
            status=response.status,
            headers=dict(response.headers),
            body=body,
            url=str(response.url),
            method=method,
            request_headers=dict(response.request_info.headers),
            from_cache=False,
        )

    def _generate_cache_key(self, method: str, url: str, kwargs: Dict[str, Any]) -> str:
        """
        Generate cache key for request.

        Args:
            method: HTTP method
            url: Request URL
            kwargs: Request arguments

        Returns:
            Cache key string
        """
        # Include method, URL, and body (for POST) in cache key
        key_parts = [method, url]

        # Include request body for POST/PUT requests
        if "json" in kwargs:
            import json

            body = json.dumps(kwargs["json"], sort_keys=True)
            key_parts.append(body)
        elif "data" in kwargs:
            # For form data, include in key
            data = kwargs["data"]
            if isinstance(data, (str, bytes)):
                key_parts.append(str(data))

        # Hash to create stable key
        key_string = ":".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"{method}:{key_hash}"

    async def _store_response_with_body(
        self,
        method: str,
        url: str,
        response: aiohttp.ClientResponse,
        cache_key: str,
        request_kwargs: Dict[str, Any],
        body: bytes,
    ) -> None:
        """
        Store response in cache with pre-read body.

        Args:
            method: HTTP method
            url: Request URL
            response: aiohttp response to cache
            cache_key: Cache key
            request_kwargs: Original request arguments
            body: Pre-read response body
        """
        # Convert aiohttp response to Hishel Entry
        hishel_request = Request(
            method=method,
            url=str(response.url),
            headers=Headers(dict(response.request_info.headers)),
            stream=None,
            metadata=request_kwargs.get("metadata", {}),
        )

        # Create async iterator factory for body (can be called multiple times)
        def body_stream_factory() -> AsyncIterator[bytes]:
            async def body_stream() -> AsyncIterator[bytes]:
                yield body

            return body_stream()

        hishel_response = Response(
            status_code=response.status,
            headers=Headers(dict(response.headers)),
            stream=body_stream_factory(),
            metadata={},
        )

        # Store in cache - this returns an Entry with wrapped stream
        entry = await self.storage.create_entry(
            hishel_request, hishel_response, cache_key
        )

        # IMPORTANT: Consume the wrapped stream to actually save to Redis
        # The storage wraps the stream with _save_stream which saves chunks as they're read
        if entry.response and entry.response.stream:
            stream = entry.response.stream
            if hasattr(stream, "__aiter__"):
                async for _ in stream:
                    pass  # Just consume, data already yielded from body_stream()
            else:
                for _ in stream:
                    pass  # Sync iterator fallback

    async def close(self) -> None:
        """Close the session and storage."""
        await self.session.close()
        await self.storage.close()

    async def __aenter__(self) -> "CachedAiohttpSession":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
