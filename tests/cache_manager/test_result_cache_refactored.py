"""
Comprehensive and refactored tests for src/cache_manager/result_cache.py

This file contains corrected tests that align with the singleton Redis client
architecture for the @cached_result decorator.
"""

import hashlib
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache_manager import result_cache
from src.cache_manager.result_cache import (
    _compute_schema_hash,
    _generate_cache_key,
    cached_result,
    close_result_cache_redis_client,
    get_result_cache_redis_client,
)


class TestRedisClientLifecycle:
    """Tests the lifecycle and configuration of the singleton Redis client."""

    @pytest.mark.asyncio
    async def test_get_client_is_singleton(self, monkeypatch):
        """Test that get_result_cache_redis_client returns a singleton."""
        monkeypatch.setattr(result_cache, "_redis_client", None)
        with patch("src.cache_manager.result_cache.Redis.from_url") as mock_from_url:
            mock_from_url.return_value = AsyncMock()
            client1 = await get_result_cache_redis_client()
            client2 = await get_result_cache_redis_client()

            assert client1 is client2
            mock_from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_uses_config_url(self, monkeypatch):
        """Test that get_result_cache_redis_client uses the URL from config."""
        monkeypatch.setattr(result_cache, "_redis_client", None)
        with patch("src.cache_manager.result_cache.Redis.from_url") as mock_from_url:
            with patch(
                "src.cache_manager.result_cache.get_cache_config"
            ) as mock_get_config:
                mock_get_config.return_value = MagicMock(
                    redis_url="redis://custom:1234/1"
                )
                await get_result_cache_redis_client()
                mock_from_url.assert_called_once_with(
                    "redis://custom:1234/1", decode_responses=True
                )

    @pytest.mark.asyncio
    async def test_get_client_uses_default_url(self, monkeypatch):
        """Test that get_result_cache_redis_client uses the default URL if config is None."""
        monkeypatch.setattr(result_cache, "_redis_client", None)
        with patch("src.cache_manager.result_cache.Redis.from_url") as mock_from_url:
            with patch(
                "src.cache_manager.result_cache.get_cache_config"
            ) as mock_get_config:
                mock_get_config.return_value = MagicMock(redis_url=None)
                await get_result_cache_redis_client()
                mock_from_url.assert_called_once_with(
                    "redis://localhost:6379/0", decode_responses=True
                )

    @pytest.mark.asyncio
    async def test_close_client(self, monkeypatch):
        """Test that close_result_cache_redis_client closes the client."""
        monkeypatch.setattr(result_cache, "_redis_client", None)
        mock_client = AsyncMock()
        with patch(
            "src.cache_manager.result_cache.Redis.from_url", return_value=mock_client
        ):
            # Get the client to initialize it
            await get_result_cache_redis_client()
            assert result_cache._redis_client is not None

            # Now close it
            await close_result_cache_redis_client()

            # Assert it was closed and the global is None
            mock_client.close.assert_called_once()
            assert result_cache._redis_client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self, monkeypatch):
        """Test that closing does nothing if client is already None."""
        monkeypatch.setattr(result_cache, "_redis_client", None)
        assert result_cache._redis_client is None
        # This should not raise any error
        await close_result_cache_redis_client()
        assert result_cache._redis_client is None


class TestCachedResultDecoratorRefactored:
    """Test cached_result() decorator with refactored singleton client logic."""

    @pytest.mark.asyncio
    async def test_decorator_basic_usage(self) -> None:
        """Test basic decorator usage with cache miss and hit."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test_data"}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis

            mock_redis.get.return_value = None
            result1 = await fetch_data("item1")

            assert result1 == {"id": "item1", "data": "test_data"}
            assert call_count == 1
            mock_redis.setex.assert_called_once()

            mock_redis.get.return_value = json.dumps(
                {"id": "item1", "data": "test_data"}
            )
            result2 = await fetch_data("item1")

            assert result2 == {"id": "item1", "data": "test_data"}
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_redis_connection_error(self) -> None:
        """Test decorator behavior when Redis connection fails."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_get_client.side_effect = Exception("Connection failed")

            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_redis_set_error(self) -> None:
        """Test decorator behavior when Redis setex() fails."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None
            mock_redis.setex.side_effect = Exception("Redis setex failed")

            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            assert call_count == 2  # Called in try, then again in except


class TestComputeSchemaHash:
    """Test _compute_schema_hash() function."""

    def test_regular_function_hash(self) -> None:
        """Test schema hash computation for regular function."""

        def test_func() -> str:
            return "test"

        hash1 = _compute_schema_hash(test_func)

        # Should be 8-character hex string
        assert len(hash1) == 8
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_same_function_same_hash(self) -> None:
        """Test that same function produces same hash."""

        def test_func() -> str:
            return "test"

        hash1 = _compute_schema_hash(test_func)
        hash2 = _compute_schema_hash(test_func)

        assert hash1 == hash2

    def test_different_functions_different_hash(self) -> None:
        """Test that different functions produce different hashes."""

        def func1() -> str:
            return "func1"

        def func2() -> str:
            return "func2"

        hash1 = _compute_schema_hash(func1)
        hash2 = _compute_schema_hash(func2)

        assert hash1 != hash2

    def test_lambda_function_uses_name_fallback(self) -> None:
        """Test that lambda functions use name-based hash fallback."""
        lambda_func = lambda x: x + 1  # noqa: E731

        hash_result = _compute_schema_hash(lambda_func)

        # Should still be 8-character hex string
        assert len(hash_result) == 8
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_builtin_function_uses_name_fallback(self) -> None:
        """Test that built-in functions use name-based hash fallback."""
        hash_result = _compute_schema_hash(len)

        # Should compute hash from function name "len"
        expected_hash = hashlib.md5("len".encode()).hexdigest()[:8]
        assert hash_result == expected_hash


class TestGenerateCacheKey:
    """Test _generate_cache_key() function."""

    def test_basic_cache_key(self) -> None:
        """Test basic cache key generation with prefix and schema hash."""
        key = _generate_cache_key("test_func", "abc12345")
        assert key == "result_cache:test_func:abc12345"

    def test_cache_key_with_string_args(self) -> None:
        """Test cache key with string arguments."""
        key = _generate_cache_key("test_func", "abc12345", "arg1", "arg2")
        assert key == "result_cache:test_func:abc12345:arg1:arg2"

    def test_cache_key_with_int_args(self) -> None:
        """Test cache key with integer arguments."""
        key = _generate_cache_key("test_func", "abc12345", 42, 100)
        assert key == "result_cache:test_func:abc12345:42:100"

    def test_cache_key_with_float_args(self) -> None:
        """Test cache key with float arguments."""
        key = _generate_cache_key("test_func", "abc12345", 3.14)
        assert key == "result_cache:test_func:abc12345:3.14"

    def test_cache_key_with_bool_args(self) -> None:
        """Test cache key with boolean arguments."""
        key = _generate_cache_key("test_func", "abc12345", True, False)
        assert "True" in key
        assert "False" in key

    def test_cache_key_with_complex_arg_dict(self) -> None:
        """Test cache key with dictionary argument (JSON serialized)."""
        key = _generate_cache_key("test_func", "abc12345", {"key": "value", "num": 42})
        assert "result_cache:test_func:abc12345:" in key
        # Should contain JSON
        assert '"key"' in key or "'key'" in key

    def test_cache_key_with_complex_arg_list(self) -> None:
        """Test cache key with list argument (JSON serialized)."""
        key = _generate_cache_key("test_func", "abc12345", [1, 2, 3])
        assert "result_cache:test_func:abc12345:" in key
        assert "[1" in key

    def test_cache_key_with_simple_kwargs(self) -> None:
        """Test cache key with simple keyword arguments."""
        key = _generate_cache_key(
            "test_func", "abc12345", param1="value1", param2=42, param3=True
        )
        assert "param1=value1" in key
        assert "param2=42" in key
        assert "param3=True" in key

    def test_cache_key_with_none_kwarg(self) -> None:
        """Test cache key with None value in kwargs."""
        key = _generate_cache_key("test_func", "abc12345", optional=None)
        assert "optional=None" in key

    def test_cache_key_with_complex_kwarg_dict(self) -> None:
        """Test cache key with complex dictionary in kwargs (JSON serialized)."""
        key = _generate_cache_key(
            "test_func",
            "abc12345",
            options={"nested": {"key": "value"}, "list": [1, 2]},
        )
        assert "options=" in key
        # Should contain JSON-serialized dict
        assert '"nested"' in key or "'nested'" in key

    def test_cache_key_with_complex_kwarg_list(self) -> None:
        """Test cache key with complex list in kwargs (JSON serialized)."""
        key = _generate_cache_key("test_func", "abc12345", items=[{"id": 1}, {"id": 2}])
        assert "items=" in key
        assert '"id"' in key or "'id'" in key

    def test_cache_key_kwargs_sorted(self) -> None:
        """Test that kwargs are sorted for stability."""
        key1 = _generate_cache_key(
            "test_func", "abc12345", z="last", a="first", m="middle"
        )
        key2 = _generate_cache_key(
            "test_func", "abc12345", a="first", m="middle", z="last"
        )
        assert key1 == key2

    def test_cache_key_long_key_hashed(self) -> None:
        """Test that very long cache keys are hashed."""
        # Create a key longer than 200 characters
        long_arg = "x" * 250
        key = _generate_cache_key("test_func", "abc12345", long_arg)

        # Should contain hash instead of full key
        assert "result_cache:test_func:abc12345:" in key
        assert len(key) < 300  # Should be hashed and shorter
        # Should contain SHA256 hash (64 hex chars)
        parts = key.split(":")
        assert len(parts) == 4
        assert len(parts[3]) == 64  # SHA256 hash

    def test_cache_key_stability_same_inputs(self) -> None:
        """Test that same inputs always produce same cache key."""
        key1 = _generate_cache_key("func", "hash123", "arg1", 42, param="value")
        key2 = _generate_cache_key("func", "hash123", "arg1", 42, param="value")
        assert key1 == key2


class TestCachedResultDecoratorExtended:
    """Extended tests for cached_result() decorator covering all edge cases."""

    @pytest.mark.asyncio
    async def test_decorator_cache_disabled_sqlite_storage(self) -> None:
        """Test decorator when storage_type is sqlite (not redis)."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch("src.cache_manager.result_cache.get_cache_config") as mock_config:
            mock_config.return_value = MagicMock(enabled=True, storage_type="sqlite")

            result1 = await fetch_data("item1")
            result2 = await fetch_data("item1")

            # Function should be called both times (caching disabled for non-redis)
            assert call_count == 2
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_decorator_uses_function_name_as_default_prefix(self) -> None:
        """Test that decorator uses function name as default key prefix."""

        @cached_result(ttl=60)
        async def my_custom_function(item_id: str) -> Dict[str, str]:
            return {"id": item_id}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            await my_custom_function("item1")

            # Check that function name is used in cache key
            cache_key = mock_redis.get.call_args[0][0]
            assert "my_custom_function" in cache_key

    @pytest.mark.asyncio
    async def test_decorator_returns_none_not_cached(self) -> None:
        """Test that None results are not cached."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Optional[Dict[str, str]]:
            nonlocal call_count
            call_count += 1
            return None

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            result1 = await fetch_data("item1")
            result2 = await fetch_data("item1")

            assert result1 is None
            assert result2 is None
            assert call_count == 2  # Called both times
            assert not mock_redis.setex.called  # Not cached

    @pytest.mark.asyncio
    async def test_decorator_with_default_ttl(self) -> None:
        """Test decorator with default TTL (no explicit ttl parameter)."""

        @cached_result(key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, str]:
            return {"id": item_id}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            await fetch_data("item1")

            # Should use default 24h TTL (86400 seconds)
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[0][1] == 86400  # Default TTL

    @pytest.mark.asyncio
    async def test_decorator_with_custom_ttl(self) -> None:
        """Test decorator with custom TTL."""

        @cached_result(ttl=3600, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, str]:
            return {"id": item_id}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            await fetch_data("item1")

            # Should use custom TTL
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[0][1] == 3600

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function name and docstring."""

        @cached_result(ttl=60, key_prefix="test")
        async def my_function(item_id: str) -> Dict[str, str]:
            """My function docstring."""
            return {"id": item_id}

        # functools.wraps should preserve metadata
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My function docstring."

    @pytest.mark.asyncio
    async def test_decorator_with_complex_return_type(self) -> None:
        """Test decorator with complex nested return types."""

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_complex_data(item_id: str) -> Dict[str, Any]:
            return {
                "id": item_id,
                "nested": {"key": "value", "list": [1, 2, 3]},
                "array": ["a", "b", "c"],
                "number": 42,
                "float": 3.14,
                "bool": True,
                "null": None,
            }

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis

            # Cache miss - store complex data
            mock_redis.get.return_value = None
            result1 = await fetch_complex_data("item1")

            # Verify complex data serialized
            assert mock_redis.setex.called
            stored_data = mock_redis.setex.call_args[0][2]
            parsed = json.loads(stored_data)
            assert parsed["nested"]["list"] == [1, 2, 3]

            # Cache hit - retrieve complex data
            mock_redis.get.return_value = stored_data
            result2 = await fetch_complex_data("item1")

            assert result1 == result2
            assert result2["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_decorator_different_args_different_cache_entries(self) -> None:
        """Test that different arguments create different cache entries."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": f"data_{item_id}"}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            result1 = await fetch_data("item1")
            result2 = await fetch_data("item2")

            assert result1["id"] == "item1"
            assert result2["id"] == "item2"
            assert call_count == 2  # Both called (different cache keys)

    @pytest.mark.asyncio
    async def test_decorator_schema_hash_included_in_key(self) -> None:
        """Test that schema hash is included in cache key."""

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, str]:
            return {"id": item_id}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            await fetch_data("item1")

            # Get the cache key used
            cache_key = mock_redis.get.call_args[0][0]

            # Cache key should have format: result_cache:prefix:schema_hash:args
            parts = cache_key.split(":")
            assert parts[0] == "result_cache"
            assert parts[1] == "test"  # prefix
            assert len(parts[2]) == 8  # schema hash (8 characters)
            assert parts[3] == "item1"  # argument
