"""
HTTP Cache Manager for enrichment pipeline.

Provides cached HTTP sessions for both aiohttp (async) and requests (sync) clients.
Supports Redis (multi-agent) and SQLite (single-agent) storage backends.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import hishel
import requests
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from .config import CacheConfig

logger = logging.getLogger(__name__)


class HTTPCacheManager:
    """Manages HTTP cache for enrichment pipeline with Redis/SQLite backends."""

    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config
        self._storage: Optional[Any] = None
        self._redis_client: Optional[Redis[bytes]] = None

        if self.config.enabled:
            self._init_storage()

    def _init_storage(self) -> None:
        """Initialize cache storage backend."""
        if self.config.storage_type == "redis":
            self._init_redis_storage()
        elif self.config.storage_type == "sqlite":
            self._init_sqlite_storage()
        else:
            raise ValueError(f"Unknown storage type: {self.config.storage_type}")

    def _init_redis_storage(self) -> None:
        """Initialize Redis storage with custom SyncRedisStorage backend."""
        try:
            if not self.config.redis_url:
                raise ValueError("redis_url required for Redis storage")

            # Test Redis connection
            self._redis_client = Redis.from_url(
                self.config.redis_url, decode_responses=False
            )
            self._redis_client.ping()

            # Import custom Redis storage implementation
            from src.cache_manager.redis_storage import SyncRedisStorage

            # Initialize custom Redis storage with default TTL
            # Note: Service-specific TTLs will be passed via request metadata
            self._storage = SyncRedisStorage(
                client=self._redis_client,
                redis_url=self.config.redis_url,
                default_ttl=86400,  # 24 hours default
                refresh_ttl_on_access=True,
                key_prefix="hishel_cache",
            )

            logger.info(
                f"Redis cache storage initialized: {self.config.redis_url} (custom backend)"
            )

        except (RedisConnectionError, ValueError, Exception) as e:
            logger.warning(
                f"Redis connection failed: {e}. Falling back to SQLite storage."
            )
            self._init_sqlite_storage()

    def _init_sqlite_storage(self) -> None:
        """Initialize SQLite file-based storage."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use Hishel 1.0 SQLite storage with absolute path
        database_path = (cache_dir / "http_cache.db").absolute()
        self._storage = hishel.SyncSqliteStorage(database_path=str(database_path))
        logger.info(f"SQLite cache initialized: {database_path}")

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
        if not self.config.enabled:
            return aiohttp.ClientSession(**session_kwargs)

        # Create async Redis storage for aiohttp caching
        try:
            from redis.asyncio import Redis as AsyncRedis

            from src.cache_manager.aiohttp_adapter import CachedAiohttpSession
            from src.cache_manager.async_redis_storage import AsyncRedisStorage

            # Create async Redis client
            async_redis_client = AsyncRedis.from_url(
                self.config.redis_url or "redis://localhost:6379/0",
                decode_responses=False,
            )

            # Get service-specific TTL
            ttl = self._get_service_ttl(service)

            # Create async storage
            async_storage = AsyncRedisStorage(
                client=async_redis_client,
                redis_url=self.config.redis_url or "redis://localhost:6379/0",
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
            logger.info(
                f"Async Redis cache initialized for {service} (TTL: {ttl}s, body-based caching: enabled)"
            )
            return CachedAiohttpSession(storage=async_storage, **session_kwargs)

        except ImportError as e:
            logger.warning(f"Async caching dependencies missing: {e}")
            return aiohttp.ClientSession(**session_kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize async cache: {e}")
            return aiohttp.ClientSession(**session_kwargs)

    def get_requests_session(
        self, service: str, **session_kwargs: Any
    ) -> requests.Session:
        """
        Get cached requests session for a service.

        Args:
            service: Service name (e.g., "jikan", "animeschedule", "anisearch")
            **session_kwargs: Additional requests.Session arguments

        Returns:
            Cached or uncached requests.Session
        """
        if not self.config.enabled or not self._storage:
            return requests.Session(**session_kwargs)

        # Get service-specific TTL
        ttl = self._get_service_ttl(service)

        # Create cached session using Hishel 1.0 CacheAdapter pattern
        try:
            from hishel.requests import CacheAdapter

            session = requests.Session(**session_kwargs)
            # Mount cache adapter for both HTTP and HTTPS
            adapter = CacheAdapter(storage=self._storage)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            return session
        except ImportError:
            logger.error(
                "hishel[requests] not installed. Install with: pip install hishel[requests]"
            )
            return requests.Session(**session_kwargs)

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

    def close(self) -> None:
        """Close cache connections."""
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")

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
            "cache_dir": (
                self.config.cache_dir if self.config.storage_type == "sqlite" else None
            ),
            "redis_url": (
                self.config.redis_url if self.config.storage_type == "redis" else None
            ),
        }
