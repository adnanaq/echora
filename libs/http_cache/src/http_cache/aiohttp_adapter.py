"""
aiohttp caching adapter using Hishel's AsyncRedisStorage.

Provides a drop-in replacement for aiohttp.ClientSession that adds
HTTP caching via Redis backend.
"""

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, cast

import aiohttp
from aiohttp import ClientResponseError, RequestInfo
from hishel import AsyncCacheProxy, CachePolicy, SpecificationPolicy
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
        headers: dict[str, str],
        body: bytes,
        url: str,
        method: str,
        request_headers: dict[str, str],
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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
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
        kwargs: dict[str, Any],
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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
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
        policy: CachePolicy | None = None,
        force_cache: bool = False,
        always_revalidate: bool = False,
        session: aiohttp.ClientSession | None = None,
        **session_kwargs: Any,
    ) -> None:
        """
        Create a CachedAiohttpSession backed by the provided AsyncBaseStorage, optionally wrapping an existing aiohttp.ClientSession.

        Parameters:
            storage (AsyncBaseStorage): Storage backend used to persist and retrieve cached HTTP entries.
            policy (Optional[CachePolicy]): Hishel policy for cache control.
            force_cache (bool): If True, forces caching of responses by injecting headers if missing.
            always_revalidate (bool): If True, always revalidates with the server.
            session (Optional[aiohttp.ClientSession]): Optional existing aiohttp session to wrap; if omitted a new session is created.
            **session_kwargs (Any): Keyword arguments forwarded to aiohttp.ClientSession when a new session is created.
        """
        self.storage = storage
        self.policy = policy or SpecificationPolicy()
        self.force_cache = force_cache
        self.always_revalidate = always_revalidate
        self.session = session or aiohttp.ClientSession(**session_kwargs)

        async def request_sender(request: Request) -> Response:
            # This sender is called by hishel when it needs a fresh response
            # Convert hishel.Request back to aiohttp call
            # Handle streaming body if present
            data = None
            if request.stream:
                collected = b"".join([chunk async for chunk in request.stream])
                data = collected

            # Capture headers from hishel Request
            headers = dict(request.headers)

            response = await self.session.request(
                method=request.method,
                url=str(request.url),
                headers=headers,
                data=data,
            )
            try:
                body = await response.read()

                async def body_stream():
                    yield body

                # Convert aiohttp.ClientResponse to hishel.Response
                res_headers = dict(response.headers)

                # Inject Cache-Control if force_cache is enabled and headers are missing
                if self.force_cache and "cache-control" not in [
                    k.lower() for k in res_headers.keys()
                ]:
                    # Default to 24h caching if forced
                    res_headers["Cache-Control"] = "public, max-age=86400"

                return Response(
                    status_code=response.status,
                    headers=Headers(res_headers),
                    stream=body_stream(),
                )
            finally:
                response.close()

        self._proxy = AsyncCacheProxy(
            request_sender=request_sender,
            storage=self.storage,
            policy=self.policy,
        )

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
        """
        # 1. Create hishel Request object
        # Handle body for POST/PUT if present in kwargs
        body = None
        if "json" in kwargs:
            body = json.dumps(kwargs["json"]).encode("utf-8")
        elif "data" in kwargs:
            data = kwargs["data"]
            if isinstance(data, str):
                body = data.encode("utf-8")
            elif isinstance(data, bytes | bytearray):
                body = bytes(data)

        async def body_stream():
            if body:
                yield body

        request_headers = dict(kwargs.get("headers", {}))
        if self.always_revalidate:
            # Force hishel to revalidate by adding no-cache to request
            request_headers["Cache-Control"] = "no-cache"

        hishel_request = Request(
            method=method,
            url=url,
            headers=Headers(request_headers),
            stream=body_stream() if body else None,
        )

        # 2. Add metadata for hishel (TTL, body-key)
        if request_headers.get("X-Hishel-Body-Key") == "true":
            hishel_request.metadata["hishel_body_key"] = True

        # 3. Delegate to hishel proxy
        hishel_response = await self._proxy.handle_request(hishel_request)

        # 4. Convert hishel.Response back to _CachedResponse
        body_chunks = []
        if hishel_response.stream:
            if hasattr(hishel_response.stream, "__aiter__"):
                async for chunk in hishel_response.stream:
                    body_chunks.append(chunk)
            else:
                body_chunks.extend(list(hishel_response.stream))
        response_body = b"".join(body_chunks)

        # Detect if it was served from cache
        from_cache = hishel_response.metadata.get("hishel_from_cache", False)

        return _CachedResponse(
            status=hishel_response.status_code,
            headers=dict(hishel_response.headers),
            body=response_body,
            url=url,
            method=method,
            request_headers=request_headers,
            from_cache=from_cache,
        )

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