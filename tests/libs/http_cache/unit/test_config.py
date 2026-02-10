"""
Comprehensive tests for src/cache/config.py

Tests cover:
- CacheConfig model instantiation with defaults
- CacheConfig model instantiation with custom values
- Field validation and constraints
- get_cache_config() with various environment variables
- All service-specific TTL configurations
- Storage type validation
- Redis URL validation
- Cache directory configuration
"""

import pytest
from http_cache.config import CacheConfig, get_cache_config
from pydantic import ValidationError


class TestCacheConfigModel:
    """Test CacheConfig Pydantic model."""

    def test_default_values(self) -> None:
        """Test CacheConfig instantiation with all default values."""
        config = CacheConfig()

        # Core settings
        assert config.cache_enable is True
        assert config.storage_type == "redis"
        assert config.redis_url == "redis://localhost:6379/0"

        # Service-specific TTLs (all should be 24 hours = 86400 seconds)
        assert config.ttl_jikan == 86400
        assert config.ttl_anilist == 86400
        assert config.ttl_anidb == 86400
        assert config.ttl_kitsu == 86400
        assert config.ttl_anime_planet == 86400
        assert config.ttl_anisearch == 86400
        assert config.ttl_animeschedule == 86400

    def test_custom_values_redis(self) -> None:
        """Test CacheConfig with custom Redis configuration."""
        config = CacheConfig(
            cache_enable=True,
            storage_type="redis",
            redis_url="redis://custom-host:6380/1",
            ttl_jikan=3600,
            ttl_anilist=7200,
        )

        assert config.cache_enable is True
        assert config.storage_type == "redis"
        assert config.redis_url == "redis://custom-host:6380/1"
        assert config.ttl_jikan == 3600
        assert config.ttl_anilist == 7200

    def test_disabled_cache(self) -> None:
        """Test CacheConfig with caching disabled."""
        config = CacheConfig(cache_enable=False)

        assert config.cache_enable is False
        # Other settings should still have defaults
        assert config.storage_type == "redis"
        assert config.redis_url == "redis://localhost:6379/0"

    def test_all_service_ttls_custom(self) -> None:
        """Test setting custom TTLs for all services."""
        custom_ttls = {
            "ttl_jikan": 1800,
            "ttl_anilist": 3600,
            "ttl_anidb": 7200,
            "ttl_kitsu": 14400,
            "ttl_anime_planet": 28800,
            "ttl_anisearch": 43200,
            "ttl_animeschedule": 86400,
        }

        config = CacheConfig(**custom_ttls)

        assert config.ttl_jikan == 1800
        assert config.ttl_anilist == 3600
        assert config.ttl_anidb == 7200
        assert config.ttl_kitsu == 14400
        assert config.ttl_anime_planet == 28800
        assert config.ttl_anisearch == 43200
        assert config.ttl_animeschedule == 86400

    def test_invalid_storage_type(self) -> None:
        """Test that invalid storage_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CacheConfig(storage_type="invalid")  # type: ignore

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert "storage_type" in str(errors[0])

    def test_invalid_cache_enable_type(self) -> None:
        """Test that invalid cache_enable type raises ValidationError."""
        with pytest.raises(ValidationError):
            CacheConfig(cache_enable="not_a_bool")  # type: ignore

    def test_invalid_ttl_type(self) -> None:
        """Test that invalid TTL type raises ValidationError."""
        with pytest.raises(ValidationError):
            CacheConfig(ttl_jikan="not_an_int")  # type: ignore

    def test_negative_ttl(self) -> None:
        """Test that negative TTL values are accepted (Pydantic allows by default)."""
        # Note: No validation prevents negative TTLs in current implementation
        config = CacheConfig(ttl_jikan=-1)
        assert config.ttl_jikan == -1

    def test_zero_ttl(self) -> None:
        """Test that zero TTL is accepted."""
        config = CacheConfig(ttl_anilist=0)
        assert config.ttl_anilist == 0

    def test_large_ttl(self) -> None:
        """Test that very large TTL values are accepted."""
        config = CacheConfig(ttl_anidb=31_536_000)  # 1 year
        assert config.ttl_anidb == 31_536_000

    def test_redis_url_formats(self) -> None:
        """Test various Redis URL formats."""
        # Standard format
        config1 = CacheConfig(redis_url="redis://localhost:6379/0")
        assert config1.redis_url == "redis://localhost:6379/0"

        # With password
        config2 = CacheConfig(redis_url="redis://:password@localhost:6379/0")
        assert config2.redis_url == "redis://:password@localhost:6379/0"

        # With username and password
        config3 = CacheConfig(redis_url="redis://user:password@host:6379/0")
        assert config3.redis_url == "redis://user:password@host:6379/0"

        # Redis Sentinel
        config4 = CacheConfig(redis_url="redis+sentinel://localhost:26379/mymaster/0")
        assert config4.redis_url == "redis+sentinel://localhost:26379/mymaster/0"


class TestGetCacheConfig:
    """Test get_cache_config() function with environment variables."""

    def test_get_cache_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_cache_config() with no environment variables (defaults)."""
        # Clear all cache-related env vars
        monkeypatch.delenv("ENABLE_HTTP_CACHE", raising=False)
        monkeypatch.delenv("REDIS_CACHE_URL", raising=False)

        config = get_cache_config()

        assert config.cache_enable is True  # Default from getenv is "true"
        assert config.storage_type == "redis"
        assert config.redis_url == "redis://localhost:6379/0"

    def test_get_cache_config_cache_enable_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with CACHE_ENABLE=true."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("CACHE_ENABLE", "true")

        config = get_cache_config()
        assert config.cache_enable is True

    def test_get_cache_config_cache_enable_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with CACHE_ENABLE=false."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("CACHE_ENABLE", "false")

        config = get_cache_config()
        assert config.cache_enable is False

    def test_get_cache_config_cache_enable_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that CACHE_ENABLE is case-insensitive."""
        get_cache_config.cache_clear()
        # Test TRUE
        monkeypatch.setenv("CACHE_ENABLE", "TRUE")
        config = get_cache_config()
        assert config.cache_enable is True

        get_cache_config.cache_clear()
        # Test False
        monkeypatch.setenv("CACHE_ENABLE", "False")
        config = get_cache_config()
        assert config.cache_enable is False

        get_cache_config.cache_clear()
        # Test TrUe
        monkeypatch.setenv("CACHE_ENABLE", "TrUe")
        config = get_cache_config()
        assert config.cache_enable is True

    def test_get_cache_config_cache_enable_invalid_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with invalid CACHE_ENABLE value raises ValidationError."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("CACHE_ENABLE", "invalid")

        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            get_cache_config()

        # Verify the error is about the 'cache_enable' field
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("cache_enable",) for error in errors)

    def test_get_cache_config_storage_type_is_redis(self) -> None:
        """Test that storage_type is always 'redis' (hardcoded, not configurable)."""
        get_cache_config.cache_clear()
        config = get_cache_config()
        assert config.storage_type == "redis"

    def test_get_cache_config_custom_redis_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with custom REDIS_URL."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_URL", "redis://prod-redis:6379/2")

        config = get_cache_config()
        assert config.redis_url == "redis://prod-redis:6379/2"

    def test_get_cache_config_all_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with all environment variables set."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("CACHE_ENABLE", "false")
        monkeypatch.setenv("REDIS_URL", "redis://custom:6380/1")

        config = get_cache_config()

        assert config.cache_enable is False
        assert config.storage_type == "redis"
        assert config.redis_url == "redis://custom:6380/1"

    def test_get_cache_config_empty_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with empty string environment variables."""
        get_cache_config.cache_clear()
        # Empty string should use defaults
        monkeypatch.setenv("REDIS_URL", "")

        config = get_cache_config()

        # Empty strings should be used as-is (not replaced with defaults)
        assert config.redis_url == ""

    def test_get_cache_config_redis_with_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_cache_config() with Redis URL containing authentication."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_URL", "redis://:mypassword@secure-redis:6379/0")

        config = get_cache_config()
        assert config.redis_url == "redis://:mypassword@secure-redis:6379/0"

    def test_get_cache_config_preserves_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_cache_config() preserves default values for unset fields."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("CACHE_ENABLE", "false")
        # Only set one env var, others should have defaults

        config = get_cache_config()

        assert config.cache_enable is False
        # These should be defaults
        assert config.storage_type == "redis"
        assert config.redis_url == "redis://localhost:6379/0"
        # Service TTLs should be defaults
        assert config.ttl_jikan == 86400
        assert config.ttl_anilist == 86400

    def test_get_cache_config_singleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_cache_config() returns cached singleton instance."""
        get_cache_config.cache_clear()

        # First call creates instance
        config1 = get_cache_config()

        # Second call returns same instance (cached)
        config2 = get_cache_config()

        assert config1 is config2

        # After cache clear, new instance is created
        get_cache_config.cache_clear()
        config3 = get_cache_config()

        assert config3 is not config1


class TestRedisConnectionPoolConfiguration:
    """Test Redis connection pool configuration via environment variables.

    These tests verify that all documented Redis configuration env vars
    are properly loaded by get_cache_config().
    """

    def test_redis_max_connections_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that REDIS_MAX_CONNECTIONS env var is loaded."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "50")

        config = get_cache_config()
        assert config.redis_max_connections == 50

    def test_redis_socket_keepalive_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that REDIS_SOCKET_KEEPALIVE env var is loaded."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_SOCKET_KEEPALIVE", "false")

        config = get_cache_config()
        assert config.redis_socket_keepalive is False

    def test_redis_socket_connect_timeout_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that REDIS_SOCKET_CONNECT_TIMEOUT env var is loaded."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_SOCKET_CONNECT_TIMEOUT", "10")

        config = get_cache_config()
        assert config.redis_socket_connect_timeout == 10

    def test_redis_socket_timeout_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that REDIS_SOCKET_TIMEOUT env var is loaded."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "20")

        config = get_cache_config()
        assert config.redis_socket_timeout == 20

    def test_redis_retry_on_timeout_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that REDIS_RETRY_ON_TIMEOUT env var is loaded."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_RETRY_ON_TIMEOUT", "false")

        config = get_cache_config()
        assert config.redis_retry_on_timeout is False

    def test_redis_health_check_interval_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that REDIS_HEALTH_CHECK_INTERVAL env var is loaded."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "60")

        config = get_cache_config()
        assert config.redis_health_check_interval == 60

    def test_all_redis_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that all Redis configuration env vars are loaded together."""
        get_cache_config.cache_clear()
        monkeypatch.setenv("REDIS_URL", "redis://prod:6379/0")
        monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "200")
        monkeypatch.setenv("REDIS_SOCKET_KEEPALIVE", "true")
        monkeypatch.setenv("REDIS_SOCKET_CONNECT_TIMEOUT", "3")
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "15")
        monkeypatch.setenv("REDIS_RETRY_ON_TIMEOUT", "true")
        monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "45")

        config = get_cache_config()

        assert config.redis_url == "redis://prod:6379/0"
        assert config.redis_max_connections == 200
        assert config.redis_socket_keepalive is True
        assert config.redis_socket_connect_timeout == 3
        assert config.redis_socket_timeout == 15
        assert config.redis_retry_on_timeout is True
        assert config.redis_health_check_interval == 45


class TestCacheConfigIntegration:
    """Integration tests for CacheConfig with realistic scenarios."""

    def test_production_redis_setup(self) -> None:
        """Test production-like Redis configuration."""
        config = CacheConfig(
            cache_enable=True,
            storage_type="redis",
            redis_url="redis://prod-redis.example.com:6379/0",
            ttl_jikan=86400,
            ttl_anilist=86400,
        )

        assert config.cache_enable is True
        assert config.storage_type == "redis"
        assert "prod-redis.example.com" in config.redis_url

    def test_disabled_cache_scenario(self) -> None:
        """Test scenario where caching is completely disabled."""
        config = CacheConfig(cache_enable=False)

        assert config.cache_enable is False
        # Backend settings should still be valid but unused
        assert config.storage_type == "redis"
        assert config.redis_url is not None

    def test_multi_service_ttl_variation(self) -> None:
        """Test realistic scenario with different TTLs per service."""
        config = CacheConfig(
            ttl_jikan=86400,  # 24 hours - frequently updated
            ttl_anilist=86400,  # 24 hours
            ttl_anidb=86400,  # 24 hours
            ttl_kitsu=86400,  # 24 hours
            ttl_anime_planet=86400,  # 24 hours
            ttl_anisearch=86400,  # 24 hours
            ttl_animeschedule=86400,  # 24 hours
        )

        # Verify all are 24 hours as per unified configuration
        assert config.ttl_jikan == 86400
        assert config.ttl_anilist == 86400
        assert config.ttl_anidb == 86400
        assert config.ttl_kitsu == 86400
        assert config.ttl_anime_planet == 86400
        assert config.ttl_anisearch == 86400
        assert config.ttl_animeschedule == 86400
