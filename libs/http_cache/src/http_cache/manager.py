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
from redis.asyncio import Redis as AsyncRedis

from .config import CacheConfig

logger = logging.getLogger(__name__)


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

        if self.config.enabled:
            self._init_storage()

    def _init_storage(self) -> None:
        """
        Selects and initializes the configured cache storage backend.

        If `storage_type` is "redis", initializes Redis storage; otherwise raises an error.

        Raises:
            ValueError: If `config.storage_type` is not a recognized backend.
        """
        if self.config.storage_type == "redis":
            self._init_redis_storage()
        else:
            raise ValueError(f"Unknown storage type: {self.config.storage_type}")

    def _init_redis_storage(self) -> None:
        """
        Validate Redis configuration and record the resulting status in logs.

        Checks that `redis_url` is present in the configured settings and logs an info message when valid. On validation failure logs a warning indicating Redis-based HTTP caching will not be used; the function does not re-raise exceptions.
        """
        try:
            if not self.config.redis_url:
                raise ValueError("redis_url required for Redis storage")

            logger.info(
                f"Redis cache configured for aiohttp sessions: {self.config.redis_url}"
            )

        except (ValueError, Exception) as e:
            logger.warning(
                f"Redis configuration failed: {e}. "
                "Async (aiohttp) requests will not be cached on Redis."
            )

    def _get_or_create_redis_client(self) -> AsyncRedis | None:
        """
        Obtain an AsyncRedis client bound to the current event loop, creating one if necessary.

        Returns:
            AsyncRedis client bound to the current event loop, or `None` if no event loop is running or Redis URL is not configured.
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
        """
        Return an aiohttp session configured for the given service, using Redis-backed body-aware caching when enabled.

        When Redis caching is active and available, returns a CachedAiohttpSession that includes the request body in the cache key so distinct request bodies produce separate cache entries. Otherwise returns a standard aiohttp.ClientSession.

        Parameters:
            service (str): Name of the service for which the session is created (used to determine TTL).
            **session_kwargs: Additional arguments forwarded to the session constructor.

        Returns:
            aiohttp.ClientSession or CachedAiohttpSession: A session object suitable for making HTTP requests; a cached session when Redis-backed caching is available, otherwise a plain aiohttp.ClientSession.
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

            # Enable body-based caching by adding X-Hishel-Body-Key header
            # This ensures POST requests (GraphQL, etc.) include body in cache key
            # Copy headers to avoid mutating caller's dict
            base_headers = session_kwargs.get("headers") or {}
            headers = dict(base_headers)
            headers["X-Hishel-Body-Key"] = "true"
            session_kwargs["headers"] = headers

            # Return cached session
            logger.debug(
                f"Async Redis cache session created for {service} (TTL: {ttl}s, event loop: {id(self._redis_event_loop)})"
            )
            return CachedAiohttpSession(storage=async_storage, **session_kwargs)

        except ImportError as e:
            logger.warning(f"Async caching dependencies missing: {e}")
            return aiohttp.ClientSession(**session_kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize async cache: {e}")
            return aiohttp.ClientSession(**session_kwargs)

    def _get_service_ttl(self, service: str) -> int:
        """
        Retrieve TTL (time-to-live) in seconds for the given service.

        Returns:
            ttl_seconds (int): TTL in seconds for the service; defaults to 86400 (24 hours) if not configured.
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
        """
        Provide a summary of the cache manager's current configuration and status.

        Returns:
            stats (Dict[str, Any]): If caching is disabled, `{"enabled": False}`.
                If enabled, a dictionary with keys:
                - `"enabled"`: `True`
                - `"storage_type"`: configured storage backend
                - `"redis_url"`: configured Redis URL (may be `None`)
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            "redis_url": self.config.redis_url,
        }
