"""
Comprehensive tests for src/cache_manager/manager.py

Tests cover:
- HTTPCacheManager initialization with Redis (async client)
- get_aiohttp_session() with body-based caching
- Service-specific TTL configuration
- Redis connection failure behavior
- Cache statistics retrieval
- Session cleanup and closing (async)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hishel import SpecificationPolicy
from http_cache.config import CacheConfig
from http_cache.manager import HTTPCacheManager


class TestHTTPCacheManagerInit:
    """Test HTTPCacheManager initialization."""

    def test_init_with_config_disabled(self) -> None:
        """Test initialization with caching disabled."""
        config = CacheConfig(enabled=False)
        manager = HTTPCacheManager(config)

        assert manager.config == config
        assert manager._async_redis_client is None
        assert isinstance(manager.policy, SpecificationPolicy)

    def test_init_with_config_enabled_redis(self) -> None:
        """Test initialization with Redis storage enabled (lazy initialization)."""
        config = CacheConfig(enabled=True, storage_type="redis")
        manager = HTTPCacheManager(config)

        assert manager.config == config
        # Redis client is lazily initialized, not created during __init__
        assert manager._async_redis_client is None
        assert manager._redis_event_loop is None
        assert isinstance(manager.policy, SpecificationPolicy)

    def test_init_redis_url_missing_logs_warning(self) -> None:
        """Test that missing redis_url logs warning and does not crash."""
        config = CacheConfig(enabled=True, storage_type="redis", redis_url=None)

        with patch("http_cache.manager.logger") as mock_logger:
            manager = HTTPCacheManager(config)
            # Should log warning and not crash - Redis client remains None
            assert manager._async_redis_client is None
            # Check for the warning message
            assert any("redis_url required" in str(call) for call in mock_logger.warning.call_args_list)

    def test_init_invalid_storage_type_raises_error(self) -> None:
        """Test that invalid storage type raises ValueError."""
        # We bypass Pydantic validation via MagicMock to test internal guard
        config = MagicMock(spec=CacheConfig)
        config.enabled = True
        config.storage_type = "invalid"
        config.force_cache = False
        config.always_revalidate = False

        with pytest.raises(ValueError, match="Unknown storage type"):
            HTTPCacheManager(config)


class TestGetAiohttpSession:
    """Test get_aiohttp_session() method."""

    def test_get_aiohttp_session_cache_disabled(self) -> None:
        """Test that regular aiohttp session is returned when cache disabled."""
        config = CacheConfig(enabled=False)
        manager = HTTPCacheManager(config)

        with patch("http_cache.manager.aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            session = manager.get_aiohttp_session("jikan")

            assert session == mock_session
            mock_session_class.assert_called_once()

    def test_get_aiohttp_session_redis_connection_failure_fallback(self) -> None:
        """Test graceful fallback to regular session when Redis connection fails."""
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            # Simulate Redis connection failure
            mock_async_redis_class.from_url.side_effect = Exception("Redis error")

            with patch("http_cache.manager.aiohttp.ClientSession") as mock_session_class:
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
        config = CacheConfig(enabled=True, storage_type="redis")

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch("http_cache.async_redis_storage.AsyncRedisStorage") as mock_async_storage:
                with patch("http_cache.aiohttp_adapter.CachedAiohttpSession") as mock_cached_session:
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
        config = CacheConfig(enabled=True, storage_type="redis", ttl_jikan=7200)

        with patch("http_cache.manager.AsyncRedis") as mock_async_redis_class:
            mock_async_redis = MagicMock()
            mock_async_redis_class.from_url.return_value = mock_async_redis

            with patch("http_cache.async_redis_storage.AsyncRedisStorage") as mock_async_storage:
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
        config = CacheConfig(enabled=True, ttl_jikan=3600)
        manager = HTTPCacheManager(config)
        assert manager._get_service_ttl("jikan") == 3600

    def test_get_service_ttl_unknown_service_default(self) -> None:
        """Test that unknown service returns default 24h TTL."""
        config = CacheConfig(enabled=True)
        manager = HTTPCacheManager(config)
        assert manager._get_service_ttl("unknown") == 86400


class TestCacheManagerClose:
    """Test close_async() method."""

    @pytest.mark.asyncio
    async def test_close_async_with_redis_client(self) -> None:
        """Test async closing manager with Redis client."""
        config = CacheConfig(enabled=True, storage_type="redis")

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
        config = CacheConfig(enabled=False)
        manager = HTTPCacheManager(config)
        assert manager.get_stats() == {"enabled": False}

    def test_get_stats_redis(self) -> None:
        """Test stats with Redis storage."""
        config = CacheConfig(enabled=True, storage_type="redis", redis_url="redis://test")
        manager = HTTPCacheManager(config)
        stats = manager.get_stats()
        assert stats["enabled"] is True
        assert stats["redis_url"] == "redis://test"