"""
HTTP Cache Manager for enrichment pipeline.

Provides cached HTTP sessions for aiohttp (async) clients.
Supports Redis storage backend for multi-agent caching.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp
from redis.asyncio import Redis as AsyncRedis

from .config import CacheConfig

logger = logging.getLogger(__name__)


class HTTPCacheManager:
    """Manages HTTP cache for enrichment pipeline with Redis backend."""

    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config
        self._async_redis_client: Optional[AsyncRedis] = None  # type: ignore[type-arg]
        self._redis_event_loop: Optional[Any] = None

        if self.config.enabled:
            self._init_storage()

    def _init_storage(self) -> None:
        """Initialize cache storage backend."""
        if self.config.storage_type == "redis":
            self._init_redis_storage()
        else:
            raise ValueError(f"Unknown storage type: {self.config.storage_type}")

    def _init_redis_storage(self) -> None:
        """
        Validate Redis configuration.

        The actual AsyncRedis client is created lazily per event loop
        in _get_or_create_redis_client() to avoid event loop conflicts.
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

    def _get_or_create_redis_client(self) -> Optional[AsyncRedis]:  # type: ignore[type-arg]
        """
        Get or create AsyncRedis client for the current event loop.

        AsyncRedis clients have internal locks that are bound to the event loop
        where they were created. This method ensures we create a new client
        for each event loop to prevent "bound to different event loop" errors.

        Returns:
            AsyncRedis client for current event loop, or None if unavailable
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
        Get cached aiohttp session for a service with body-based caching.

        Body-based caching automatically includes request body (GraphQL queries,
        POST data, etc.) in the cache key. This ensures different queries/variables
        get different cache entries, enabling automatic cache invalidation when
        you modify API requests.

        Args:
            service: Service name (e.g., "jikan", "anilist", "anidb")
            **session_kwargs: Additional aiohttp.ClientSession arguments

        Returns:
            Cached aiohttp.ClientSession (wrapped with Redis caching)
        """
        if not self.config.enabled or self.config.storage_type != "redis":
            return aiohttp.ClientSession(**session_kwargs)

        # Get or create Redis client for current event loop
        redis_client = self._get_or_create_redis_client()
        if not redis_client:
            return aiohttp.ClientSession(**session_kwargs)

        # Create async Redis storage for aiohttp caching
        try:
            from src.cache_manager.aiohttp_adapter import CachedAiohttpSession
            from src.cache_manager.async_redis_storage import AsyncRedisStorage

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
            headers = session_kwargs.get("headers", {})
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
        Get TTL for a specific service.

        Args:
            service: Service name

        Returns:
            TTL in seconds
        """
        ttl_attr = f"ttl_{service}"
        return getattr(self.config, ttl_attr, 86400)  # Default 24 hours

    async def close_async(self) -> None:
        """Close async cache connections."""
        if self._async_redis_client:
            try:
                await self._async_redis_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async Redis client: {e}")
            finally:
                self._async_redis_client = None
                self._redis_event_loop = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            "redis_url": self.config.redis_url,
        }
