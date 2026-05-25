"""
Unit tests for http_cache.manager.

Tests cover:
- HTTPCacheManager initialization (policy, filters, Redis client ownership)
- get_aiohttp_session(): cached vs uncached, service TTL, Redis failure fallback
- Service-specific TTL lookup
- Async session and Redis client cleanup
- Cache statistics
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hishel import FilterPolicy
from http_cache.config import CacheConfig
from http_cache.exceptions import StorageConfigurationError
from http_cache.manager import HTTPCacheManager, NeverCacheErrorsFilter


class TestHTTPCacheManagerInit:
    """Test HTTPCacheManager initialization."""

    def test_init_with_config_disabled(self) -> None:
        """Test initialization with caching disabled."""
        config = CacheConfig(cache_enabled=False)
        manager = HTTPCacheManager(config)

        assert manager.config == config
        assert manager._async_redis_client is None
        assert isinstance(manager.policy, FilterPolicy)
        # Policy must NOT set use_body_key globally — body-key is opted in
        # per-request via X-Hishel-Body-Key header (GraphQL/POST only).
        assert manager.policy.use_body_key is False
        # Policy should have error filter to prevent caching errors
        assert len(manager.policy.response_filters) == 1
        assert isinstance(manager.policy.response_filters[0], NeverCacheErrorsFilter)

    def test_init_with_config_enabled_redis(self) -> None:
        """Test initialization with Redis storage enabled (lazy initialization)."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")
        manager = HTTPCacheManager(config)

        assert manager.config == config
        # Redis client is lazily initialized, not created during __init__
        assert manager._async_redis_client is None
        assert manager._redis_event_loop is None
        assert isinstance(manager.policy, FilterPolicy)
        # Policy must NOT set use_body_key globally — body-key is opted in
        # per-request via X-Hishel-Body-Key header (GraphQL/POST only).
        assert manager.policy.use_body_key is False
        # Policy should have error filter to prevent caching errors
        assert len(manager.policy.response_filters) == 1
        assert isinstance(manager.policy.response_filters[0], NeverCacheErrorsFilter)

    def test_init_redis_url_missing_logs_warning(self) -> None:
        """Test that missing redis_url logs warning and does not crash."""
        config = CacheConfig(cache_enabled=True, storage_type="redis", redis_url=None)

        with patch("http_cache.manager.logger") as mock_logger:
            manager = HTTPCacheManager(config)
            # Should log warning and not crash - Redis client remains None
            assert manager._async_redis_client is None
            # Check for the warning message
            assert any(
                "redis_url required" in str(call)
                for call in mock_logger.warning.call_args_list
            )

    def test_init_invalid_storage_type_raises_error(self) -> None:
        """Test that invalid storage type raises ValueError."""
        # We bypass Pydantic validation via MagicMock to test internal guard
        config = MagicMock(spec=CacheConfig)
        config.cache_enabled = True
        config.storage_type = "invalid"
        config.force_cache = False
        config.always_revalidate = False

        with pytest.raises(StorageConfigurationError, match="Unknown storage type"):
            HTTPCacheManager(config)


class TestGetAiohttpSession:
    """Test get_aiohttp_session() method."""

    def test_get_aiohttp_session_cache_disabled(self) -> None:
        """Test that regular aiohttp session is returned when cache disabled."""
        config = CacheConfig(cache_enabled=False)
        manager = HTTPCacheManager(config)

        with patch("http_cache.manager.aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            session = manager.get_aiohttp_session("jikan")

            assert session == mock_session
            mock_session_class.assert_called_once()

    def test_get_aiohttp_session_redis_connection_failure_fallback(self) -> None:
        """Test graceful fallback to regular session when Redis connection fails."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            # Simulate Redis connection failure
            mock_async_redis_class.from_url.side_effect = Exception("Redis error")

            with patch(
                "http_cache.manager.aiohttp.ClientSession"
            ) as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                manager = HTTPCacheManager(config)
                session = manager.get_aiohttp_session("jikan")

                # Should fall back to regular session
                assert session == mock_session
                assert manager._async_redis_client is None

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_with_redis_success(self) -> None:
        """Test aiohttp session creation with Redis caching."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "http_cache.async_redis_storage.AsyncRedisStorage"
            ) as mock_async_storage:
                with patch(
                    "http_cache.aiohttp_adapter.CachedAiohttpSession"
                ) as mock_cached_session:
                    mock_session_instance = MagicMock()
                    mock_cached_session.return_value = mock_session_instance

                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")

                    # Should create cached session
                    mock_cached_session.assert_called_once()
                    mock_async_storage.assert_called_once()
                    assert session == mock_session_instance

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_service_specific_ttl(self) -> None:
        """Test that service-specific TTL is used."""
        config = CacheConfig(cache_enabled=True, storage_type="redis", ttl_jikan=7200)

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "http_cache.async_redis_storage.AsyncRedisStorage"
            ) as mock_async_storage:
                with patch("http_cache.aiohttp_adapter.CachedAiohttpSession"):
                    manager = HTTPCacheManager(config)
                    manager.get_aiohttp_session("jikan")

                    # Check TTL passed to AsyncRedisStorage
                    call_kwargs = mock_async_storage.call_args[1]
                    assert call_kwargs["default_ttl"] == 7200.0


class TestServiceTTL:
    """Test _get_service_ttl() method."""

    def test_get_service_ttl_known_service(self) -> None:
        """Test TTL retrieval for known services."""
        config = CacheConfig(cache_enabled=True, ttl_jikan=3600)
        manager = HTTPCacheManager(config)
        assert manager._get_service_ttl("jikan") == 3600

    def test_get_service_ttl_unknown_service_default(self) -> None:
        """Test that unknown service returns default 24h TTL."""
        config = CacheConfig(cache_enabled=True)
        manager = HTTPCacheManager(config)
        assert manager._get_service_ttl("unknown") == 86400


class TestCacheManagerClose:
    """Test close_async() method."""

    @pytest.mark.asyncio
    async def test_close_async_with_redis_client(self) -> None:
        """Test async closing manager with Redis client."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch("http_cache.async_redis_storage.AsyncRedisStorage"):
                with patch("http_cache.aiohttp_adapter.CachedAiohttpSession"):
                    manager = HTTPCacheManager(config)
                    manager.get_aiohttp_session("jikan")
                    await manager.close_async()

                    mock_async_redis.aclose.assert_called_once()


class TestGetStats:
    """Test get_stats() method."""

    def test_get_stats_disabled(self) -> None:
        """Test stats when caching is disabled."""
        config = CacheConfig(cache_enabled=False)
        manager = HTTPCacheManager(config)
        assert manager.get_stats() == {"cache_enabled": False}

    def test_get_stats_redis(self) -> None:
        """Test stats with Redis storage."""
        config = CacheConfig(
            cache_enabled=True, storage_type="redis", redis_url="redis://test"
        )
        manager = HTTPCacheManager(config)
        stats = manager.get_stats()
        assert stats["cache_enabled"] is True
        assert stats["redis_url"] == "redis://test"


class TestNeverCacheErrorsFilter:
    """Test NeverCacheErrorsFilter methods directly."""

    def test_needs_body_returns_false(self) -> None:
        """needs_body() always returns False — status code alone decides cacheability."""
        f = NeverCacheErrorsFilter()
        assert f.needs_body() is False

    def test_apply_caches_success(self) -> None:
        """apply() returns True for 2xx/3xx responses."""
        from unittest.mock import MagicMock

        f = NeverCacheErrorsFilter()
        for status in (200, 201, 301, 304):
            item = MagicMock()
            item.status_code = status
            assert f.apply(item, None) is True

    def test_apply_blocks_errors(self) -> None:
        """apply() returns False for 4xx/5xx responses."""
        from unittest.mock import MagicMock

        f = NeverCacheErrorsFilter()
        for status in (400, 401, 404, 429, 500, 503):
            item = MagicMock()
            item.status_code = status
            assert f.apply(item, None) is False


class TestCloseAsyncException:
    """Test close_async() error handling path."""

    @pytest.mark.asyncio
    async def test_close_async_logs_exception_on_aclose_failure(self) -> None:
        """Exception during aclose() is logged as warning; client refs are cleared."""
        config = CacheConfig(cache_enabled=False)
        manager = HTTPCacheManager(config)

        mock_client = AsyncMock()
        mock_client.aclose.side_effect = RuntimeError("connection lost")
        manager._async_redis_client = mock_client
        manager._redis_event_loop = object()

        with patch("http_cache.manager.logger") as mock_logger:
            await manager.close_async()

        assert manager._async_redis_client is None
        assert manager._redis_event_loop is None
        assert any(
            "Error closing async Redis client" in str(call)
            for call in mock_logger.warning.call_args_list
        )


class TestGetOrCreateRedisClient:
    """Test _get_or_create_redis_client() edge cases."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_redis_url(self) -> None:
        """Returns None if redis_url is unset (L282 guard)."""
        config = CacheConfig(cache_enabled=True, storage_type="redis", redis_url=None)
        manager = HTTPCacheManager(config)
        # Simulate being inside a running event loop but with no URL
        result = manager._get_or_create_redis_client()
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_new_client_when_loop_changes(self) -> None:
        """When the stored event loop differs from current, creates a new client (L262-276)."""
        import asyncio

        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_new = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_new

            manager = HTTPCacheManager(config)

            # Simulate a first client created for a fake "old" event loop
            old_loop = MagicMock()
            old_loop.is_running.return_value = False
            manager._async_redis_client = AsyncMock()  # old client, distinct object
            manager._redis_event_loop = old_loop

            # Now call inside real running loop — loop mismatch triggers switch
            client = manager._get_or_create_redis_client()

            assert client is mock_new
            assert manager._redis_event_loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_creates_new_client_when_old_loop_still_running(self) -> None:
        """When old event loop is still running, uses run_coroutine_threadsafe (L266)."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_new = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_new

            with patch(
                "http_cache.manager.asyncio.run_coroutine_threadsafe"
            ) as mock_threadsafe:
                manager = HTTPCacheManager(config)

                old_loop = MagicMock()
                old_loop.is_running.return_value = True  # old loop still alive
                manager._async_redis_client = AsyncMock()
                manager._redis_event_loop = old_loop

                client = manager._get_or_create_redis_client()

                assert client is mock_new
                mock_threadsafe.assert_called_once()

    @pytest.mark.asyncio
    async def test_old_client_cleanup_exception_swallowed(self) -> None:
        """Exception during old-client cleanup is caught and logged (L275-276)."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_new = AsyncMock()
            mock_async_redis_class.from_url.return_value = mock_new

            manager = HTTPCacheManager(config)

            old_loop = MagicMock()
            old_loop.is_running.side_effect = RuntimeError("loop gone")
            manager._async_redis_client = AsyncMock()
            manager._redis_event_loop = old_loop

            # Should not raise despite cleanup failure
            client = manager._get_or_create_redis_client()
            assert client is mock_new

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_import_error_fallback(self) -> None:
        """ImportError in get_aiohttp_session() falls back to plain aiohttp.ClientSession."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "http_cache.async_redis_storage.AsyncRedisStorage",
                side_effect=ImportError("missing dep"),
            ):
                with patch(
                    "http_cache.manager.aiohttp.ClientSession"
                ) as mock_session_class:
                    mock_session_class.return_value = MagicMock()
                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")
                    assert session == mock_session_class.return_value

    @pytest.mark.asyncio
    async def test_get_aiohttp_session_generic_exception_fallback(self) -> None:
        """Generic Exception in get_aiohttp_session() falls back to plain aiohttp.ClientSession."""
        config = CacheConfig(cache_enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch(
                "http_cache.async_redis_storage.AsyncRedisStorage",
                side_effect=RuntimeError("init failed"),
            ):
                with patch(
                    "http_cache.manager.aiohttp.ClientSession"
                ) as mock_session_class:
                    mock_session_class.return_value = MagicMock()
                    manager = HTTPCacheManager(config)
                    session = manager.get_aiohttp_session("jikan")
                    assert session == mock_session_class.return_value
