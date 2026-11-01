"""Cached HTTP sessions and the Redis clients behind every cache."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import replace
from typing import Any

import aiohttp
from common.config import CacheConfig, RedisConfig
from common.redis_client import create_redis_client
from hishel import AsyncRedisStorage, BaseFilter, Entry, FilterPolicy, Headers, Request
from hishel import Response as HishelResponse
from redis.asyncio import Redis

from .aiohttp_adapter import CachedAiohttpSession
from .network_request_pacer import NetworkRequestPacer
from .utils import _mask_url_credentials

logger = logging.getLogger(__name__)


# Some APIs report failure in the body while still answering HTTP 200. AniDB
# does this: a rate-limit ban comes back as 200 with
# ``<error code="500">banned</error>``, which a status check alone would cache
# and keep serving long after the ban lifts.
_XML_ERROR_ROOT = re.compile(rb"^\s*(?:<\?xml[^>]*\?>\s*)?<error[\s>]", re.IGNORECASE)

# Error envelopes are tiny; cap the scan so a large body is never searched.
_MAX_ERROR_BODY_BYTES = 4096

# Request headers that carry credentials, in lower case as hishel keeps them.
_CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})


class CredentialFreeRedisStorage(AsyncRedisStorage):
    """Hishel Redis storage that writes no credential header of a request.

    Hishel stores each cached request next to its response, headers included,
    so an app token sent as ``Authorization`` would stay in Redis as long as
    the entry. Hishel has no option to leave headers out, so the stored copy
    of the request drops them. The request sent over the network keeps them.

    A cache entry is still found by its URL alone, so responses fetched with
    and without a token share an entry.
    """

    async def create_entry(
        self,
        request: Request,
        response: HishelResponse,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """Store a response with a copy of its request that holds no credentials.

        Args:
            request: The request the response answers.
            response: The response to store.
            key: The cache key hishel computed for the request.
            id_: The entry's id, or None for a new one.

        Returns:
            The stored entry.
        """
        return await super().create_entry(
            _remove_credential_headers(request), response, key, id_
        )


def _remove_credential_headers(request: Request) -> Request:
    """Return a request without its credential headers, or the request itself when it has none.

    Args:
        request: The request to store.

    Returns:
        A copy holding every other header with all its values, or the same
        request when no credential header is present, so most requests are
        never copied.
    """
    if _CREDENTIAL_HEADERS.isdisjoint(request.headers):
        return request
    return replace(
        request,
        headers=Headers(
            {
                name: request.headers.get_list(name) or []
                for name in request.headers
                if name not in _CREDENTIAL_HEADERS
            }
        ),
    )


class NeverCacheErrorsFilter(BaseFilter[HishelResponse]):
    """Hishel response filter that prevents caching of error responses.

    Rejects 4xx/5xx by status, and also rejects a 200 whose body is an XML
    error envelope, so an API that signals failure in the body cannot poison
    the cache. Forcing a fresh request matters for:

    - Rate limit errors (429, or AniDB's 200 + ``<error code="500">banned``)
    - Server errors (5xx): transient failures shouldn't be cached
    - Auth failures (401/403): auth state may change between requests
    - Not Found (404): resources might be created later

    The filter is integrated into Hishel's FilterPolicy and evaluated before
    responses are stored in the cache backend.

    Examples:
        >>> policy = FilterPolicy(response_filters=[NeverCacheErrorsFilter()])
    """

    def needs_body(self) -> bool:
        """Indicate whether response body is needed for filtering decision.

        Returns:
            True - a 200 can still carry an error envelope in its body.
        """
        return True

    def apply(self, item: HishelResponse, body: bytes | None) -> bool:
        """Determine whether a response may be cached.

        Args:
            item: Hishel Response object containing status code and headers.
            body: Response body bytes, or None when unavailable.

        Returns:
            True to allow caching, False to prevent it.

        Examples:
            >>> filter = NeverCacheErrorsFilter()
            >>> filter.apply(Response(status_code=200), b"<anime>...</anime>")
            True
            >>> filter.apply(Response(status_code=200), b'<error code="500">banned</error>')
            False
        """
        if item.status_code >= 400:
            return False
        if (
            body is not None
            and len(body) <= _MAX_ERROR_BODY_BYTES
            and _XML_ERROR_ROOT.match(body)
        ):
            return False
        return True


class HTTPCacheManager:
    """Hands out cached HTTP sessions and owns the Redis clients behind them.

    Two clients share one Redis server: one for HTTP responses, stored as raw
    bytes, and one for crawl results, stored as JSON text. A client belongs to
    the event loop it was created on, so both are created again when called
    from a different loop.

    Caching is off when ``cache.cache_enabled`` is false or no ``redis_url`` is
    configured; sessions are then plain ``aiohttp.ClientSession`` objects.
    """

    def __init__(self, cache: CacheConfig, redis: RedisConfig) -> None:
        """Set up the manager; Redis is not contacted until the first request.

        Args:
            cache: What to cache and for how long.
            redis: Where Redis is and how to size its connection pools.
        """
        self.cache = cache
        self.redis = redis
        # Body-key caching is opted in per-request via X-Hishel-Body-Key header (GraphQL/POST).
        # Enabling use_body_key globally causes all GET requests (no body) to hash to
        # SHA256(b"") — the same key — collapsing every URL into one cache slot.
        self.policy = FilterPolicy(response_filters=[NeverCacheErrorsFilter()])
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._response_client: Redis | None = None
        self._result_client: Redis | None = None
        self._network_request_pacer = NetworkRequestPacer()

        if cache.cache_enabled and not redis.redis_url:
            logger.warning(
                "Caching is enabled but REDIS_URL is not set; running uncached"
            )
        elif self.enabled and redis.redis_url:
            logger.info(f"Caching on Redis at {_mask_url_credentials(redis.redis_url)}")

    @property
    def network_request_pacer(self) -> NetworkRequestPacer:
        """The gap kept between this process's network requests to each provider.

        Every session this manager hands out, and every ``@cached_result``
        function on a cache miss, waits its turn here, so cache hits never wait.
        """
        return self._network_request_pacer

    @property
    def enabled(self) -> bool:
        """Whether responses and crawl results are cached."""
        return self.cache.cache_enabled and bool(self.redis.redis_url)

    def get_aiohttp_session(
        self, service: str, **session_kwargs: Any
    ) -> aiohttp.ClientSession | CachedAiohttpSession:
        """Return an HTTP session for a source, cached when caching is on.

        Args:
            service: The source, which sets how long responses are kept, such
                as ``"anilist"``.
            **session_kwargs: Passed to the session, such as ``timeout`` or
                ``headers``.

        Returns:
            A ``CachedAiohttpSession`` on Redis, or a plain
            ``aiohttp.ClientSession`` when caching is off or no event loop
            is running.

        Examples:
            >>> manager = HTTPCacheManager(cache_config, redis_config)
            >>> async with manager.get_aiohttp_session("anilist") as session:
            ...     async with session.post(url, json=payload) as response:
            ...         data = await response.json()
        """
        # The pacer's hook runs only for requests that reach the network.
        session_kwargs["trace_configs"] = [
            *session_kwargs.get("trace_configs", []),
            self._network_request_pacer.build_trace_config(service),
        ]
        client = self._client_for_loop(decode_responses=False)
        if client is None:
            return aiohttp.ClientSession(**session_kwargs)
        ttl = self.cache.ttl_for(service)
        storage = CredentialFreeRedisStorage(
            client=client, ttl=ttl, key_prefix="hishel_cache"
        )
        return CachedAiohttpSession(
            storage=storage,
            policy=self.policy,
            force_cache=self.cache.force_cache,
            always_revalidate=self.cache.always_revalidate,
            default_ttl=ttl,
            **session_kwargs,
        )

    def get_result_client(self) -> Redis | None:
        """Return the Redis client for crawl results, which stores JSON text.

        Returns:
            The client, or None when caching is off or no event loop is running.
        """
        return self._client_for_loop(decode_responses=True)

    def get_stats(self) -> dict[str, Any]:
        """Describe whether caching is on and where it stores entries.

        Returns:
            ``{"cache_enabled": False}`` when caching is off, otherwise
            ``cache_enabled``, ``storage_type`` and ``redis_url`` with its
            password masked.
        """
        if not self.enabled or not self.redis.redis_url:
            return {"cache_enabled": False}
        return {
            "cache_enabled": True,
            "storage_type": self.cache.storage_type,
            "redis_url": _mask_url_credentials(self.redis.redis_url),
        }

    async def close(self) -> None:
        """Close both Redis clients; later requests create new ones."""
        clients = [self._response_client, self._result_client]
        self._event_loop = self._response_client = self._result_client = None
        for client in clients:
            if client is None:
                continue
            try:
                await client.aclose()
            except Exception as error:
                # Best effort: a failed close must not stop the app shutting down.
                logger.warning(f"Error closing Redis client: {error}")

    def _client_for_loop(self, *, decode_responses: bool) -> Redis | None:
        """Return a client bound to the running event loop, creating it if needed.

        Args:
            decode_responses: True for the result client, False for the
                response client.

        Returns:
            The client, or None when caching is off or no event loop is running.
        """
        if not self.enabled:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        if self._event_loop is not loop:
            self._release_clients()
            self._event_loop = loop
        if decode_responses:
            if self._result_client is None:
                self._result_client = create_redis_client(
                    self.redis, decode_responses=True
                )
            return self._result_client
        if self._response_client is None:
            self._response_client = create_redis_client(self.redis)
        return self._response_client

    def _release_clients(self) -> None:
        """Drop the clients of a previous event loop.

        A client can only be closed on its own loop. When that loop still runs
        in another thread the close is sent there; a finished loop already
        dropped its connections.
        """
        old_loop = self._event_loop
        for client in (self._response_client, self._result_client):
            if client is None or old_loop is None or not old_loop.is_running():
                continue
            close = client.aclose()
            try:
                asyncio.run_coroutine_threadsafe(close, old_loop)
            except RuntimeError as error:
                # The loop stopped after the check; its connections went with it.
                close.close()
                logger.debug(f"Could not close previous Redis client: {error}")
        self._response_client = self._result_client = None
