"""Aiohttp caching adapter with RFC 9111-compliant HTTP caching via Hishel.

This module provides a drop-in replacement for aiohttp.ClientSession that adds
intelligent HTTP caching with Redis backend. Key features:

- **RFC 9111 Compliance**: Respects Cache-Control, ETag, Last-Modified, Expires
- **Body-Aware Caching**: Different POST bodies produce separate cache entries
  (critical for GraphQL APIs)
- **Error Filtering**: Never caches 4xx/5xx responses (rate limits, server errors)
- **Reusable Streams**: Solves stream exhaustion for body-key caching
- **Auto Content-Type**: Automatically injects Content-Type header for JSON requests

Architecture:
    The adapter wraps aiohttp.ClientSession and integrates with Hishel's
    AsyncCacheProxy, which implements RFC 9111 caching logic. The flow is:

    1. Application calls session.get() or session.post()
    2. CachedAiohttpSession converts to Hishel Request with ReusableBodyStream
    3. Hishel proxy checks cache (reads stream for cache key if body-key enabled)
    4. On cache miss, calls request_sender() callback (reads stream again)
    5. request_sender() uses aiohttp to make actual HTTP request
    6. Response flows back through Hishel (caches if cacheable) to application

Examples:
    Basic usage with GET request::

        from http_cache.instance import http_cache_manager

        session = http_cache_manager.get_aiohttp_session("jikan")
        async with session.get("https://api.jikan.moe/v4/anime/21") as response:
            data = await response.json()
            print(response.from_cache)  # False on first request, True on second

    GraphQL POST request with body-aware caching::

        session = http_cache_manager.get_aiohttp_session("anilist")
        query = "query ($id: Int!) { Media(id: $id) { title { romaji } } }"

        # First request - cache miss
        async with session.post(
            "https://graphql.anilist.co",
            json={"query": query, "variables": {"id": 171018}}
        ) as response:
            data = await response.json()  # from_cache=False

        # Same request - cache hit (body included in cache key)
        async with session.post(
            "https://graphql.anilist.co",
            json={"query": query, "variables": {"id": 171018}}
        ) as response:
            data = await response.json()  # from_cache=True

        # Different variables - cache miss (different cache key)
        async with session.post(
            "https://graphql.anilist.co",
            json={"query": query, "variables": {"id": 21}}
        ) as response:
            data = await response.json()  # from_cache=False

Note:
    This module requires Hishel >=1.1.9 for body-key caching support via
    FilterPolicy.use_body_key attribute.
"""

import json
import logging
from types import TracebackType
from typing import Any, cast

import aiohttp
from aiohttp import ClientResponseError, RequestInfo
from hishel import AsyncBaseStorage, AsyncCacheProxy, CachePolicy, FilterPolicy
from hishel._core._headers import Headers
from hishel._core.models import Request, Response
from multidict import CIMultiDict, CIMultiDictProxy  # pants: no-infer-dep
from yarl import URL  # pants: no-infer-dep

logger = logging.getLogger(__name__)


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
    """Aiohttp session wrapper with RFC 9111-compliant HTTP caching via Hishel.

    This class provides a drop-in replacement for aiohttp.ClientSession that adds
    intelligent HTTP caching. It wraps Hishel's AsyncCacheProxy and implements the
    request_sender pattern to integrate aiohttp with Hishel's caching logic.

    The caching flow:
        1. Application calls session.get() or session.post()
        2. Request converted to Hishel Request with ReusableBodyStream
        3. Hishel proxy checks cache (computes cache key including body if enabled)
        4. Cache hit: Returns cached response immediately
        5. Cache miss: Calls request_sender() to fetch from origin
        6. request_sender() uses wrapped aiohttp session for actual HTTP request
        7. Response stored in cache (if cacheable) and returned to application

    Attributes:
        storage: Hishel storage backend (typically AsyncRedisStorage).
        policy: Hishel caching policy (typically FilterPolicy with body-key enabled).
        force_cache: Whether to override bad/missing cache headers.
        always_revalidate: Whether to force revalidation on every request.
        session: Underlying aiohttp.ClientSession for actual HTTP requests.

    Examples:
        Basic usage::

            storage = AsyncRedisStorage(redis_client)
            session = CachedAiohttpSession(storage=storage)

            async with session.get("https://api.example.com/data") as response:
                data = await response.json()
                print(response.from_cache)  # True/False

        With force_cache for APIs with bad headers::

            session = CachedAiohttpSession(
                storage=storage,
                force_cache=True  # Override no-cache headers
            )

        Custom policy::

            policy = FilterPolicy(response_filters=[CustomFilter()])
            policy.use_body_key = True
            session = CachedAiohttpSession(storage=storage, policy=policy)
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
        """Initialize cached aiohttp session with Hishel integration.

        Creates a CachedAiohttpSession that wraps an aiohttp.ClientSession and
        integrates with Hishel's AsyncCacheProxy for RFC 9111-compliant caching.

        The request_sender callback is defined here to bridge Hishel and aiohttp:
        - Hishel calls request_sender when it needs a fresh response
        - request_sender converts Hishel Request to aiohttp request
        - Reads body from ReusableBodyStream
        - Makes actual HTTP request via wrapped aiohttp session
        - Converts aiohttp response to Hishel Response
        - Optionally injects cache headers if force_cache=True

        Args:
            storage: Hishel storage backend for cache persistence (e.g.,
                AsyncRedisStorage, AsyncFileStorage).
            policy: Hishel caching policy. If None, creates FilterPolicy with
                use_body_key=True for GraphQL support.
            force_cache: If True, overrides missing/bad cache headers by injecting
                Cache-Control with configured TTL. Useful for APIs that don't send
                proper cache headers.
            always_revalidate: If True, adds Cache-Control: no-cache to all
                requests, forcing revalidation with server.
            session: Existing aiohttp.ClientSession to wrap. If None, creates new
                session with session_kwargs.
            **session_kwargs: Additional arguments forwarded to aiohttp.ClientSession
                constructor (e.g., timeout, headers, connector).

        Note:
            The request_sender async function is defined as a closure within __init__
            to access self.session and self.force_cache. It's passed to Hishel's
            AsyncCacheProxy and called on cache misses.
        """
        self.storage = storage
        # Use FilterPolicy with body-key enabled for GraphQL/POST caching
        if policy is None:
            policy = FilterPolicy()
            policy.use_body_key = True
        self.policy = policy
        self.force_cache = force_cache
        self.always_revalidate = always_revalidate
        self.session = session or aiohttp.ClientSession(**session_kwargs)
        # Snapshot default headers so per-request calls can merge them.
        # (aiohttp stores defaults on the session object; tests may pass an AsyncMock.)
        try:
            self._session_default_headers: dict[str, str] = dict(  # type: ignore[arg-type]
                getattr(self.session, "headers", {})  # aiohttp: CIMultiDictProxy
            )
        except Exception:
            self._session_default_headers = {}

        async def request_sender(request: Request) -> Response:
            """Hishel callback to fetch fresh responses from origin server.

            This callback bridges Hishel's caching logic with aiohttp's HTTP client.
            Hishel calls this function when:
            - Cache miss (no cached entry found)
            - Cached entry is stale and needs revalidation
            - Cache-Control: no-cache forces fresh request

            The function:
            1. Reads body from request.stream (ReusableBodyStream handles multiple reads)
            2. Filters out X-Hishel-* control headers
            3. Makes actual HTTP request via wrapped aiohttp session
            4. Optionally injects cache headers if force_cache=True
            5. Converts aiohttp response to Hishel Response format

            Args:
                request: Hishel Request object with method, url, headers, and stream.

            Returns:
                Hishel Response object with status, headers, and body stream.

            Note:
                This is a closure that accesses self.session and self.force_cache
                from the enclosing CachedAiohttpSession.__init__ scope.
            """
            # Handle streaming body if present
            data = None
            if request.stream:
                collected = b"".join([chunk async for chunk in request.stream])
                data = collected

            # Capture headers from hishel Request.
            # Do not forward X-Hishel-* control headers upstream.
            headers = {
                k: v
                for k, v in dict(request.headers).items()
                if not k.lower().startswith("x-hishel-")
            }

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

                # Force caching: override upstream cache headers so Hishel's RFC state machine
                # will treat responses as fresh cacheable content. This is necessary for APIs
                # that default to `Cache-Control: no-cache` or omit caching headers entirely.
                #
                # IMPORTANT: Never cache error responses (4xx/5xx) as they are often temporary:
                # - 429 (Rate Limit): Retrying should hit API, not cached error
                # - 500/502/503: Transient server errors should not be cached
                # - 401/403: Auth errors should not be cached
                if self.force_cache and 200 <= response.status < 400:
                    ttl = None
                    try:
                        ttl_meta = request.metadata.get("hishel_ttl")  # type: ignore[union-attr]
                        if isinstance(ttl_meta, int | float):
                            ttl = int(ttl_meta)
                    except Exception:
                        ttl = None
                    if ttl is None:
                        # Fall back to storage default TTL when available.
                        storage_ttl = getattr(self.storage, "default_ttl", None)
                        if isinstance(storage_ttl, int | float):
                            ttl = int(storage_ttl)
                    if ttl is None:
                        ttl = 86400

                    # Replace/define Cache-Control. Also strip legacy/contradictory headers.
                    res_headers["Cache-Control"] = f"public, max-age={ttl}"
                    res_headers.pop("Pragma", None)
                    res_headers.pop("Expires", None)
                elif response.status >= 400:
                    # Explicitly prevent caching of error responses
                    res_headers["Cache-Control"] = "no-store"

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

    @staticmethod
    def _parse_hishel_bool(value: str) -> bool | None:
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return None

    @staticmethod
    def _extract_hishel_metadata_and_strip_headers(
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """
        Extract Hishel control values from request headers and remove them from headers.

        Hishel supports controlling caching via metadata (preferred) or X-Hishel-* headers
        in some integrations. Our aiohttp adapter translates those headers to metadata and
        strips them so we don't leak control headers to upstream servers or pollute vary.
        """
        metadata: dict[str, Any] = {}

        # Work case-insensitively but preserve original dict.
        lower_to_key = {k.lower(): k for k in list(headers.keys())}

        def pop_ci(name: str) -> str | None:
            key = lower_to_key.get(name.lower())
            if key is None:
                return None
            return headers.pop(key, None)

        ttl_raw = pop_ci("X-Hishel-Ttl")
        if ttl_raw is not None:
            try:
                metadata["hishel_ttl"] = float(ttl_raw)
            except ValueError:
                pass

        refresh_raw = pop_ci("X-Hishel-Refresh-Ttl-On-Access")
        if refresh_raw is not None:
            parsed = CachedAiohttpSession._parse_hishel_bool(refresh_raw)
            if parsed is not None:
                metadata["hishel_refresh_ttl_on_access"] = parsed

        spec_ignore_raw = pop_ci("X-Hishel-Spec-Ignore")
        if spec_ignore_raw is not None:
            parsed = CachedAiohttpSession._parse_hishel_bool(spec_ignore_raw)
            if parsed is not None:
                metadata["hishel_spec_ignore"] = parsed

        body_key_raw = pop_ci("X-Hishel-Body-Key")
        if body_key_raw is not None:
            parsed = CachedAiohttpSession._parse_hishel_bool(body_key_raw)
            if parsed is True:
                metadata["hishel_body_key"] = True

        return metadata

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
        # 1. Compute effective URL (include params in cache key and origin request).
        request_url = URL(url)
        params = kwargs.get("params")
        if params is not None:
            request_url = request_url.update_query(params)

        # 2. Merge headers: session defaults + per-request headers.
        request_headers = {
            **self._session_default_headers,
            **dict(kwargs.get("headers", {})),
        }

        # Add Content-Type header for JSON requests
        if "json" in kwargs and "content-type" not in {
            k.lower() for k in request_headers
        }:
            request_headers["Content-Type"] = "application/json"

        # 3. Extract X-Hishel-* control headers to metadata and strip from request headers
        # This prevents control headers from being sent to upstream and affecting cache keys
        hishel_metadata: dict[str, Any] = (
            self._extract_hishel_metadata_and_strip_headers(request_headers)
        )

        # 4. Create hishel Request object (including body when present).
        # Handle body for POST/PUT if present in kwargs.
        body: bytes | None = None
        unsupported_body = False
        if "json" in kwargs:
            body = json.dumps(kwargs["json"]).encode("utf-8")
        elif "data" in kwargs:
            data = kwargs["data"]
            if isinstance(data, str):
                body = data.encode("utf-8")
            elif isinstance(data, bytes | bytearray):
                body = bytes(data)
            elif data is not None:
                # We can't reliably serialize arbitrary aiohttp data objects (FormData, streams, etc.).
                # Avoid breaking semantics by bypassing cache in that case.
                unsupported_body = True

        # Create a reusable stream that can be iterated multiple times.
        # This is critical because Hishel will:
        # 1. Read the stream to compute cache key (when use_body_key=True)
        # 2. Pass the request to request_sender which needs to read stream again
        # Using a class with __aiter__ creates a fresh iterator each time.
        from collections.abc import AsyncIterator as AsyncIteratorABC

        class ReusableBodyStream(AsyncIteratorABC[bytes]):
            """Reusable async stream for HTTP request bodies.

            This class solves the stream exhaustion problem that occurs when Hishel
            needs to read the request body multiple times:

            1. First read: Hishel computes cache key (when use_body_key=True)
            2. Second read: request_sender needs body to send to upstream server

            A standard async generator can only be iterated once. By implementing
            __aiter__, this class creates a fresh iterator on each iteration,
            allowing the same body data to be read multiple times.

            Attributes:
                body_data: The request body bytes, or None for GET/HEAD requests.

            Examples:
                >>> stream = ReusableBodyStream(b'{"query": "..."}')
                >>> # First iteration (Hishel computes cache key)
                >>> async for chunk in stream:
                ...     print(len(chunk))  # 15
                >>> # Second iteration (request_sender sends to server)
                >>> async for chunk in stream:
                ...     print(len(chunk))  # 15 (same data!)

            Note:
                This is an inner class defined in _request() method scope to access
                the body variable. It's instantiated once per request.
            """

            def __init__(self, body_data: bytes | None):
                """Initialize reusable stream with request body data.

                Args:
                    body_data: Request body bytes, or None for bodyless requests.
                """
                self.body_data = body_data

            def __aiter__(self):
                """Create a fresh async iterator for the body data.

                Returns:
                    Async generator that yields body_data once if not None.

                Note:
                    This method is called each time the stream is iterated,
                    ensuring a fresh iterator every time.
                """

                async def _gen():
                    if self.body_data is not None:
                        yield self.body_data

                return _gen()

            async def __anext__(self) -> bytes:
                """Compatibility method for AsyncIterator protocol."""
                raise StopAsyncIteration

        if self.always_revalidate:
            # Force hishel to revalidate by adding no-cache to request
            request_headers["Cache-Control"] = "no-cache"

        # When body-key is enabled, Hishel requires a stream even for GET requests
        # Provide reusable stream for requests (None body for GET, actual body for POST)
        needs_stream = body is not None or self.policy.use_body_key

        # If we can't safely represent the body in the cache key or upstream request,
        # fall back to a non-cached network request.
        if unsupported_body:
            async with self.session.request(
                method=method,
                url=str(request_url),
                headers=request_headers,
                params=None,  # already encoded into URL
                **{k: v for k, v in kwargs.items() if k not in ("headers", "params")},
            ) as resp:
                raw = await resp.read()
                return _CachedResponse(
                    status=resp.status,
                    headers=dict(resp.headers),
                    body=raw,
                    url=str(request_url),
                    method=method,
                    request_headers=request_headers,
                    from_cache=False,
                )

        hishel_request = Request(
            method=method,
            url=str(request_url),
            headers=Headers(request_headers),
            stream=ReusableBodyStream(body) if needs_stream else None,
            metadata=hishel_metadata,
        )

        # 5. Delegate to hishel proxy
        hishel_response = await self._proxy.handle_request(hishel_request)

        # 6. Convert hishel.Response back to _CachedResponse
        body_chunks = []
        if hishel_response.stream:
            if hasattr(hishel_response.stream, "__aiter__"):
                # Cast to AsyncIterator after runtime check
                from collections.abc import AsyncIterator as AsyncIteratorType

                stream = cast(AsyncIteratorType[bytes], hishel_response.stream)
                async for chunk in stream:
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
            url=str(request_url),
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
