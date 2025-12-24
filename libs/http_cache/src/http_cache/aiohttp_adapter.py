"""
aiohttp caching adapter using Hishel's AsyncRedisStorage.

Provides a drop-in replacement for aiohttp.ClientSession that adds
HTTP caching via Redis backend.

**Note**: This is a simplified implementation that directly uses
AsyncRedisStorage. For async helpers (AniList, Kitsu, AniDB), we use
the cached session which wraps the original aiohttp.ClientSession.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Optional, Type, cast

import aiohttp
from aiohttp import ClientResponseError, RequestInfo
from hishel._core._headers import Headers
from hishel._core._storages._async_base import AsyncBaseStorage
from hishel._core.models import Request, Response
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


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
        Initialize a lightweight cached response that mimics the minimal aiohttp.ClientResponse interface.
        
        Parameters:
            status (int): HTTP status code of the response.
            headers (Dict[str, str]): Response headers; stored case-insensitively.
            body (bytes): Raw response body bytes.
            url (str): Request URL associated with the response.
            method (str): HTTP method used for the request (e.g., "GET", "POST").
            request_headers (Dict[str, str]): Headers from the original request.
            from_cache (bool): Whether this response was served from cache.
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
        """
        Retrieve the cached response body.
        
        Returns:
            bytes: Raw response body bytes.
        """
        return self._body

    async def text(self, encoding: str = "utf-8") -> str:
        """
        Decode the cached response body to a text string.
        
        Parameters:
            encoding (str): Character encoding used to decode the response body. Defaults to "utf-8".
        
        Returns:
            str: The response body decoded as text.
        """
        return self._body.decode(encoding)

    async def json(
        self,
    ) -> Any:
        """
        Parse the response body as JSON.
        
        Returns:
            The Python object produced by decoding the response body with JSON.
        """

        return json.loads(self._body.decode("utf-8"))

    def release(self) -> None:
        """
        Mark the cached response as released.
        
        Sets the internal released flag so the response is treated as closed; this does not affect network resources or perform I/O.
        """
        self._released = True

    def raise_for_status(self) -> None:
        """
        Raise a ClientResponseError if the response status indicates a client or server error.
        
        Raises:
            aiohttp.ClientResponseError: when the response status is between 400 and 599 inclusive,
            populated with request URL, method, headers, status, message, and response headers.
        """
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
        """
        Enter the async context and yield the cached response instance.
        
        Returns:
            _CachedResponse: the context-managed cached response object.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the async context for the response and release held resources.
        
        Called when leaving an `async with` block; invokes `release()` to free the underlying response and does not suppress exceptions raised within the context.
        """
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
        Initialize the async context manager for a pending HTTP request.
        
        Parameters:
            coro: Coroutine that yields an aiohttp.ClientResponse when awaited.
            session: Parent CachedAiohttpSession that created this context manager.
            method: HTTP method for the request (e.g., "GET", "POST").
            url: Request URL.
            kwargs: Keyword arguments passed to the underlying request (headers, params, json, etc.).
        """
        self._coro = coro
        self._session = session
        self._method = method
        self._url = url
        self._kwargs = kwargs
        self._response: _CachedResponse | None = None

    async def __aenter__(self) -> "_CachedResponse":
        """
        Execute the pending request and return the resulting cached or live response.
        
        Also stores the obtained response on the context manager instance.
        
        Returns:
            _CachedResponse: The response object produced by the request.
        """
        self._response = await self._coro
        return cast(_CachedResponse, self._response)

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the async context and release the underlying response if present.
        
        If a response was obtained during __aenter__, calls its `release()` method to free resources.
        """
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
        Create a CachedAiohttpSession backed by the provided AsyncBaseStorage, optionally wrapping an existing aiohttp.ClientSession.
        
        Parameters:
            storage (AsyncBaseStorage): Storage backend used to persist and retrieve cached HTTP entries.
            session (Optional[aiohttp.ClientSession]): Optional existing aiohttp session to wrap; if omitted a new session is created.
            **session_kwargs (Any): Keyword arguments forwarded to aiohttp.ClientSession when a new session is created.
        """
        self.storage = storage
        self.session = session or aiohttp.ClientSession(**session_kwargs)

    def get(self, url: str, **kwargs: Any) -> _CachedRequestContextManager:
        """
        Create an async context manager that performs a GET request and serves cached responses when available.
        
        Parameters:
            url (str): Request URL.
            **kwargs: Additional aiohttp request keyword arguments (e.g., params, headers, json).
        
        Returns:
            _CachedRequestContextManager: Async context manager that yields a `_CachedResponse` for the GET request.
        """
        coro = self._request("GET", url, **kwargs)
        return _CachedRequestContextManager(coro, self, "GET", url, kwargs)

    def post(self, url: str, **kwargs: Any) -> _CachedRequestContextManager:
        """
        Create an async context manager for a cached POST request to the given URL.
        
        Parameters:
            url (str): The request URL.
            **kwargs: Additional aiohttp request arguments that may affect the request or cache key (for example `json`, `params`, `data`, `headers`).
        
        Returns:
            _CachedRequestContextManager: An async context manager that yields a cached `_CachedResponse` when available or performs the POST and returns a live `_CachedResponse` on cache miss.
        """
        coro = self._request("POST", url, **kwargs)
        return _CachedRequestContextManager(coro, self, "POST", url, kwargs)

    async def _request(self, method: str, url: str, **kwargs: Any) -> "_CachedResponse":
        """
        Perform an HTTP request using the wrapped session and return a cached-friendly response, serving from Redis-backed storage when a matching cache entry exists.
        
        Parameters:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): Request URL.
            **kwargs: Additional arguments forwarded to the underlying aiohttp session request (headers, params, json, data, etc.).
        
        Returns:
            _CachedResponse: A response wrapper containing status, headers, body, url, method, and request headers. The wrapper's `from_cache` flag is `True` when the response was served from storage and `False` when it was fetched live.
        
        Notes:
            - On a cache hit this function reconstructs the response body and headers from the most recently created cache entry and does not perform a network request.
            - On a cache miss it performs the network request, reads the full body, and stores the response in storage only if the HTTP status is less than 400.
        """
        # Generate cache key
        cache_key = self._generate_cache_key(method, url, kwargs)

        # Check cache
        entries = await self.storage.get_entries(cache_key)
        if entries:
            # Cache hit - return cached response WITHOUT making HTTP request
            # Select entry with latest created_at timestamp (not entries[0])
            # Redis SMEMBERS returns unordered results, so entries[0] may be stale
            entry = max(
                entries,
                key=lambda cached_entry: getattr(
                    cached_entry.meta, "created_at", 0.0
                ) if cached_entry.meta else 0.0,
            )
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
                        # Iterator
                        body_chunks.extend(list(stream))
                body = b"".join(body_chunks)

                # Extract headers - convert to simple dict
                if isinstance(entry.response.headers, Headers):
                    # Headers._headers is list-based with nested structure
                    # Convert to dict (taking last value for duplicate keys)
                    headers_dict: Dict[str, str] = {}
                    try:
                        # Try direct iteration first
                        for item in entry.response.headers._headers:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                key, value = item[0], item[1]
                                headers_dict[key] = value
                    except (ValueError, TypeError):
                        # Fallback: headers might be already a dict
                        headers_dict = dict(entry.response.headers._headers)  # type: ignore[arg-type]  # _headers is private multidict format
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
                method, response, cache_key, kwargs, body
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
        Create a deterministic cache key for an HTTP request.
        
        Parameters:
            method (str): HTTP method (e.g., "GET", "POST").
            url (str): Request URL.
            kwargs (Dict[str, Any]): Request keyword arguments; `json`, `params`, and `data`
                values (when present) are included in a deterministic serialized form.
        
        Returns:
            str: Cache key in the form "<METHOD>:<sha256 hex>", derived from the method,
                 URL, and serialized payloads to ensure stable, comparable keys.
        """

        key_parts = [method, url]

        def serialize_payload(value: Any) -> str:
            """
            Create a deterministic string representation of arbitrary payloads suitable for cache key generation.
            
            Parameters:
                value (Any): The payload to serialize; may be bytes, str, mappings, sequences, sets, or other objects.
            
            Returns:
                serialized (str): A stable string form of `value`. Bytes/bytearray are decoded as UTF-8 (ignoring errors); mappings are converted to a JSON-encoded list of key/value pairs with keys sorted and values recursively serialized; sequences and sets are JSON-encoded lists of recursively serialized items; all other values use `str(value)`.
            """
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8", errors="ignore")
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                items = sorted(
                    (str(key), serialize_payload(item_value))
                    for key, item_value in value.items()
                )
                return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (Sequence, set)) and not isinstance(
                value, (bytes, bytearray, str)
            ):
                serialized = [serialize_payload(item) for item in value]
                return json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
            return str(value)

        if "json" in kwargs:
            key_parts.append(
                json.dumps(
                    kwargs["json"],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

        if params := kwargs.get("params"):
            key_parts.append(f"params={serialize_payload(params)}")

        if "data" in kwargs:
            key_parts.append(f"data={serialize_payload(kwargs['data'])}")

        # Hash to create stable key
        key_string = ":".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"{method}:{key_hash}"

    async def _store_response_with_body(
        self,
        method: str,
        response: aiohttp.ClientResponse,
        cache_key: str,
        request_kwargs: Dict[str, Any],
        body: bytes,
    ) -> None:
        """
        Store an aiohttp response and its pre-read body into the storage backend under the given cache key.

        Converts the live aiohttp response and request information into Hishel `Request` and `Response`
        models, creates a cache entry via the storage backend, and consumes the entry's response stream
        so any wrapped stream persistence logic (e.g., saving chunks to Redis) is triggered.

        Parameters:
            method: HTTP method used for the original request.
            response: The aiohttp.ClientResponse that was received.
            cache_key: The key under which the entry will be stored in the backend.
            request_kwargs: Original request keyword arguments; `metadata` (if present) is propagated to the stored Request.
            body: The full response body already read from `response`, provided as bytes.
        """
        # Convert aiohttp response to Hishel Entry
        hishel_request = Request(
            method=method,
            url=str(response.url),
            headers=Headers(dict(response.request_info.headers)),
            stream=None,  # type: ignore[arg-type]  # Request body not needed for caching
            metadata=request_kwargs.get("metadata", {}),
        )

        # Create async iterator factory for body (can be called multiple times)
        def body_stream_factory() -> AsyncIterator[bytes]:
            """
            Create an async iterator that yields the captured response body as a single bytes chunk.
            
            Returns:
                AsyncIterator[bytes]: An async iterator which yields one `bytes` value containing the full body.
            """
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
        """
        Enter the async context for the cached aiohttp session.
        
        Returns:
            CachedAiohttpSession: The same session instance to be used within the async with block.
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """
        Exit the async context manager and close managed resources.
        
        Called when leaving an `async with` block; ensures the session and associated storage are closed.
        """
        await self.close()
