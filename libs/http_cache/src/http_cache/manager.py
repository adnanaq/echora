"""
HTTP Cache Manager for enrichment pipeline.

Provides cached HTTP sessions for aiohttp (async) clients.
Supports Redis storage backend for multi-agent caching.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from hishel import BaseFilter, FilterPolicy
from hishel._core.models import Response as HishelResponse
from redis.asyncio import Redis as AsyncRedis

from .config import CacheConfig
from .exceptions import StorageConfigurationError

logger = logging.getLogger(__name__)


class NeverCacheErrorsFilter(BaseFilter[HishelResponse]):
    """Hishel response filter that prevents caching of HTTP error responses.

    This filter ensures that error responses (4xx/5xx status codes) are never
    cached, forcing fresh network requests on retry. This is critical for:

    - Rate limit errors (429): Retries should check current limit status
    - Server errors (5xx): Transient failures shouldn't be cached
    - Auth failures (401/403): Auth state may change between requests
    - Not Found (404): Resources might be created later

    The filter is integrated into Hishel's FilterPolicy and evaluated before
    responses are stored in the cache backend.

    Examples:
        >>> policy = FilterPolicy(response_filters=[NeverCacheErrorsFilter()])
        >>> manager = HTTPCacheManager(policy=policy)

    Note:
        This filter works at the Hishel policy level, preventing error responses
        from ever reaching the storage backend. It's more efficient than storing
        and checking errors later.
    """

    def needs_body(self) -> bool:
        """Indicate whether response body is needed for filtering decision.

        Returns:
            Always False - status code alone determines cacheability.
        """
        return False

    def apply(self, item: HishelResponse, body: bytes | None) -> bool:
        """Determine if response should be cached based on status code.

        Args:
            item: Hishel Response object containing status code and headers.
            body: Response body bytes (unused, always None since needs_body=False).

        Returns:
            True to allow caching (2xx/3xx), False to prevent caching (4xx/5xx).

        Examples:
            >>> filter = NeverCacheErrorsFilter()
            >>> response_200 = Response(status_code=200, ...)
            >>> filter.apply(response_200, None)  # Returns True (cache allowed)
            True
            >>> response_429 = Response(status_code=429, ...)
            >>> filter.apply(response_429, None)  # Returns False (no cache)
            False
        """
        # Only cache successful responses (2xx, 3xx)
        return item.status_code < 400


class HTTPCacheManager:
    """Manages HTTP cache for enrichment pipeline with Redis backend."""

    def __init__(self, config: CacheConfig):
        """
        Initialize the HTTPCacheManager with the provided cache configuration.

        Parameters:
            config (CacheConfig): Configuration that controls whether caching is enabled and specifies storage backend and Redis-related settings. If `config.enabled` is true, storage backend initialization is attempted.
        """
        self.config = config
        self._async_redis_client: AsyncRedis | None = None
        self._redis_event_loop: Any | None = None
        # Enable body-key caching for POST requests (GraphQL queries)
        # Body-key ensures different queries/variables in POST body have separate cache entries
        # Add response filter to prevent caching of error responses
        self.policy = FilterPolicy(response_filters=[NeverCacheErrorsFilter()])
        self.policy.use_body_key = True  # Include request body in cache key

        if self.config.enabled:
            self._init_storage()

    def _init_storage(self) -> None:
        """
        Selects and initializes the configured cache storage backend.

        If `storage_type` is "redis", initializes Redis storage; otherwise raises an error.

        Raises:
            StorageConfigurationError: If `config.storage_type` is not a recognized backend.
        """
        if self.config.storage_type == "redis":
            self._init_redis_storage()
        else:
            raise StorageConfigurationError(self.config.storage_type)

    def _init_redis_storage(self) -> None:
        """
        Validate Redis configuration and record the resulting status in logs.

        Checks that `redis_url` is present in the configured settings and logs an info message when valid. On validation failure logs a warning indicating Redis-based HTTP caching will not be used; the function does not re-raise exceptions.
        """
        # Validate configuration before logging
        if not self.config.redis_url:
            # Configuration error - log warning instead of failing
            logger.warning(
                "Redis configuration failed: redis_url required for Redis storage. "
                "Async (aiohttp) requests will not be cached on Redis."
            )
            return

        # Configuration valid - log success
        logger.info(
            f"Redis cache configured for aiohttp sessions: {self.config.redis_url}"
        )

    def _get_or_create_redis_client(self) -> AsyncRedis | None:
        """Obtain an AsyncRedis client bound to the current event loop.

        Creates a new client if necessary or reuses existing client for current loop.
        This ensures each asyncio event loop has its own Redis connection, which is
        critical for multi-agent concurrent processing.

        Returns:
            AsyncRedis client bound to current event loop, or None if no event loop
            is running or Redis URL is not configured.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running
            return None

        # Check if we need to create a new client for this event loop
        if self._async_redis_client is None or self._redis_event_loop != current_loop:
            # Close old client if it exists
            if self._async_redis_client is not None:
                old_client = self._async_redis_client
                old_loop = self._redis_event_loop
                try:
                    if old_loop and old_loop.is_running():
                        asyncio.run_coroutine_threadsafe(old_client.aclose(), old_loop)
                    else:
                        cleanup_task = current_loop.create_task(old_client.aclose())
                        cleanup_task.add_done_callback(
                            lambda t: logger.debug(
                                "Old Redis client cleanup completed: %s",
                                t.exception() if t.exception() else "success",
                            )
                        )
                except Exception as close_error:
                    logger.debug(
                        "Failed to close previous Redis client: %s", close_error
                    )

            # Create new client for current event loop
            if not self.config.redis_url:
                return None

            # Configure connection pool for multi-agent concurrency and reliability
            self._async_redis_client = AsyncRedis.from_url(
                self.config.redis_url,
                decode_responses=False,
                max_connections=self.config.redis_max_connections,
                socket_keepalive=self.config.redis_socket_keepalive,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                socket_timeout=self.config.redis_socket_timeout,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                health_check_interval=self.config.redis_health_check_interval,
            )
            self._redis_event_loop = current_loop
            logger.debug(
                f"Created new AsyncRedis client for event loop {id(current_loop)} "
                f"(max_connections={self.config.redis_max_connections})"
            )

        return self._async_redis_client

    def get_aiohttp_session(
        self, service: str, **session_kwargs: Any
    ) -> Any:  # Returns aiohttp.ClientSession or CachedAiohttpSession
        """Get an aiohttp session for the specified service with caching support.

        When Redis caching is enabled and available, returns a CachedAiohttpSession
        with body-aware caching (critical for GraphQL/POST requests). Different
        request bodies produce separate cache entries. Falls back to standard
        aiohttp.ClientSession if caching is disabled or unavailable.

        Args:
            service: Service name used to determine cache TTL (e.g., "anilist", "jikan").
            **session_kwargs: Additional arguments forwarded to session constructor
                (e.g., timeout, headers, connector).

        Returns:
            CachedAiohttpSession with Redis backend if caching enabled, otherwise
            plain aiohttp.ClientSession.

        Examples:
            >>> manager = HTTPCacheManager(config)
            >>> session = manager.get_aiohttp_session("anilist")
            >>> async with session.post(url, json=payload) as response:
            ...     data = await response.json()
        """
        if not self.config.enabled or self.config.storage_type != "redis":
            return aiohttp.ClientSession(**session_kwargs)

        # Get or create Redis client for current event loop
        redis_client = self._get_or_create_redis_client()
        if not redis_client:
            return aiohttp.ClientSession(**session_kwargs)

        # Create async Redis storage for aiohttp caching
        try:
            from http_cache.aiohttp_adapter import CachedAiohttpSession
            from http_cache.async_redis_storage import AsyncRedisStorage

            # Get service-specific TTL
            ttl = self._get_service_ttl(service)

            # Create async storage from event-loop-specific client
            async_storage = AsyncRedisStorage(
                client=redis_client,
                default_ttl=float(ttl),
                refresh_ttl_on_access=True,
                key_prefix="hishel_cache",
            )

            # Return cached session
            logger.debug(
                f"Async Redis cache session created for {service} (TTL: {ttl}s, event loop: {id(self._redis_event_loop)})"
            )
            return CachedAiohttpSession(
                storage=async_storage,
                policy=self.policy,
                force_cache=self.config.force_cache,
                always_revalidate=self.config.always_revalidate,
                **session_kwargs,
            )

        except ImportError as e:
            logger.warning(f"Async caching dependencies missing: {e}")
            return aiohttp.ClientSession(**session_kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize async cache: {e}")
            return aiohttp.ClientSession(**session_kwargs)

    def _get_service_ttl(self, service: str) -> int:
        """Retrieve cache TTL in seconds for the specified service.

        Args:
            service: Service name (e.g., "anilist", "jikan", "kitsu").

        Returns:
            TTL in seconds for the service, defaults to 86400 (24 hours) if not
            configured via CacheConfig.ttl_{service}.
        """
        ttl_attr = f"ttl_{service}"
        return getattr(self.config, ttl_attr, 86400)  # Default 24 hours

    async def close_async(self) -> None:
        """
        Close and cleanup the asynchronous Redis client used for HTTP caching.

        If an async Redis client exists, attempts to close it and logs any exception raised during closure.
        Afterward, clears internal references to the Redis client and its associated event loop.
        """
        if self._async_redis_client:
            try:
                await self._async_redis_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async Redis client: {e}")
            finally:
                self._async_redis_client = None
                self._redis_event_loop = None

    def get_stats(self) -> dict[str, Any]:
        """Get summary of cache manager's current configuration and status.

        Returns:
            Dictionary with cache configuration. If caching disabled, returns
            {"enabled": False}. If enabled, includes:
            - "enabled": True
            - "storage_type": Backend type (e.g., "redis")
            - "redis_url": Redis connection URL (may be None)
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            "redis_url": self.config.redis_url,
        }
