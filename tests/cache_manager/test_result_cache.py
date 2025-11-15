"""
Comprehensive tests for src/cache_manager/result_cache.py

Tests cover:
- _compute_schema_hash() with regular functions, lambdas, built-ins
- _generate_cache_key() with various argument combinations
- cached_result() decorator with cache hits/misses
- Schema hash invalidation when function code changes
- Redis connection handling and error cases
- TTL configuration
- Cache key stability and uniqueness
- Error handling and fallback behavior
"""

import hashlib
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache_manager.result_cache import (
    _compute_schema_hash,
    _generate_cache_key,
    cached_result,
)


class TestComputeSchemaHash:
    """Test _compute_schema_hash() function."""

    def test_regular_function_hash(self) -> None:
        """Test schema hash computation for regular function."""

        def test_func() -> str:
            return "test"

        hash1 = _compute_schema_hash(test_func)

        # Should be 16-character hex string (64 bits for collision resistance)
        assert len(hash1) == 16
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

    def test_function_code_change_changes_hash(self) -> None:
        """Test that changing function code changes hash."""

        def func_v1() -> str:
            return "version 1"

        hash1 = _compute_schema_hash(func_v1)

        def func_v2() -> str:
            return "version 2"

        hash2 = _compute_schema_hash(func_v2)

        # Different code = different hash
        assert hash1 != hash2

    def test_lambda_function_uses_name_fallback(self) -> None:
        """Test that lambda functions use name-based hash fallback."""
        # Lambda source code can't be retrieved, so uses function name
        lambda_func = lambda x: x + 1  # noqa: E731

        hash_result = _compute_schema_hash(lambda_func)

        # Should still be 16-character hex string
        assert len(hash_result) == 16
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_builtin_function_uses_name_fallback(self) -> None:
        """Test that built-in functions use name-based hash fallback."""
        # Built-in functions don't have source code
        hash_result = _compute_schema_hash(len)

        # Should compute hash from function name "len" (16 characters)
        expected_hash = hashlib.md5("len".encode()).hexdigest()[:16]
        assert hash_result == expected_hash

    def test_function_with_multiline_code(self) -> None:
        """Test hash computation for functions with multiline code."""

        def complex_func(x: int, y: int) -> int:
            """Complex function with multiple lines."""
            result = x + y
            if result > 10:
                return result * 2
            return result

        hash_result = _compute_schema_hash(complex_func)

        assert len(hash_result) == 16
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_function_with_decorators(self) -> None:
        """Test hash computation for decorated functions."""

        def decorator(func):  # type: ignore
            def wrapper(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)

            return wrapper

        @decorator
        def decorated_func() -> str:
            return "decorated"

        hash_result = _compute_schema_hash(decorated_func)

        # Should hash the wrapper function's source
        assert len(hash_result) == 16


class TestGenerateCacheKey:
    """Test _generate_cache_key() function."""

    def test_basic_cache_key(self) -> None:
        """Test basic cache key generation with prefix and schema hash."""
        key = _generate_cache_key("test_func", "abc12345")

        assert key == "result_cache:test_func:abc12345"

    def test_cache_key_with_single_arg(self) -> None:
        """Test cache key with single string argument."""
        key = _generate_cache_key("test_func", "abc12345", "arg1")

        assert key == "result_cache:test_func:abc12345:arg1"

    def test_cache_key_with_multiple_args(self) -> None:
        """Test cache key with multiple arguments."""
        key = _generate_cache_key("test_func", "abc12345", "arg1", "arg2", "arg3")

        assert key == "result_cache:test_func:abc12345:arg1:arg2:arg3"

    def test_cache_key_with_int_arg(self) -> None:
        """Test cache key with integer argument."""
        key = _generate_cache_key("test_func", "abc12345", 42)

        assert key == "result_cache:test_func:abc12345:42"

    def test_cache_key_with_float_arg(self) -> None:
        """Test cache key with float argument."""
        key = _generate_cache_key("test_func", "abc12345", 3.14)

        assert key == "result_cache:test_func:abc12345:3.14"

    def test_cache_key_with_bool_arg(self) -> None:
        """Test cache key with boolean argument."""
        key1 = _generate_cache_key("test_func", "abc12345", True)
        key2 = _generate_cache_key("test_func", "abc12345", False)

        assert key1 == "result_cache:test_func:abc12345:True"
        assert key2 == "result_cache:test_func:abc12345:False"

    def test_cache_key_with_dict_arg(self) -> None:
        """Test cache key with dictionary argument (JSON serialized)."""
        key = _generate_cache_key("test_func", "abc12345", {"key": "value"})

        assert "result_cache:test_func:abc12345:" in key
        assert '{"key": "value"}' in key or '{"key":"value"}' in key

    def test_cache_key_with_list_arg(self) -> None:
        """Test cache key with list argument (JSON serialized)."""
        key = _generate_cache_key("test_func", "abc12345", [1, 2, 3])

        assert "result_cache:test_func:abc12345:" in key
        assert "[1, 2, 3]" in key or "[1,2,3]" in key

    def test_cache_key_with_kwargs(self) -> None:
        """Test cache key with keyword arguments."""
        key = _generate_cache_key(
            "test_func", "abc12345", param1="value1", param2="value2"
        )

        # Kwargs should be sorted alphabetically
        assert "param1=value1" in key
        assert "param2=value2" in key

    def test_cache_key_with_none_kwarg(self) -> None:
        """Test cache key with None value in kwargs."""
        key = _generate_cache_key("test_func", "abc12345", optional=None)

        assert "optional=None" in key

    def test_cache_key_kwargs_sorted(self) -> None:
        """Test that kwargs are sorted for stability."""
        key1 = _generate_cache_key(
            "test_func", "abc12345", z="last", a="first", m="middle"
        )
        key2 = _generate_cache_key(
            "test_func", "abc12345", a="first", m="middle", z="last"
        )

        # Keys should be identical regardless of kwargs order
        assert key1 == key2

    def test_cache_key_with_complex_dict_sorted(self) -> None:
        """Test that dict keys are sorted in JSON serialization."""
        key = _generate_cache_key("test_func", "abc12345", {"z": 1, "a": 2, "m": 3})

        # JSON should have sorted keys
        assert json.dumps({"z": 1, "a": 2, "m": 3}, sort_keys=True) in key

    def test_cache_key_long_key_hashed(self) -> None:
        """Test that very long cache keys are hashed."""
        # Create a key longer than 200 characters
        long_arg = "x" * 200

        key = _generate_cache_key("test_func", "abc12345", long_arg)

        # Should contain hash instead of full key
        assert "result_cache:test_func:abc12345:" in key
        assert len(key) < 250  # Should be hashed and shorter

    def test_cache_key_stability_same_inputs(self) -> None:
        """Test that same inputs always produce same cache key."""
        key1 = _generate_cache_key("func", "hash123", "arg1", 42, param="value")
        key2 = _generate_cache_key("func", "hash123", "arg1", 42, param="value")

        assert key1 == key2

    def test_cache_key_different_for_different_schema_hash(self) -> None:
        """Test that different schema hashes produce different cache keys."""
        key1 = _generate_cache_key("func", "hash111", "arg1")
        key2 = _generate_cache_key("func", "hash222", "arg1")

        assert key1 != key2
        assert "hash111" in key1
        assert "hash222" in key2

    def test_cache_key_with_complex_kwarg_dict(self) -> None:
        """Test cache key with complex dictionary in kwargs (JSON serialized)."""
        key = _generate_cache_key(
            "test_func",
            "abc12345",
            options={"nested": {"key": "value"}, "list": [1, 2]},
        )

        # Should contain JSON-serialized dict
        assert "result_cache:test_func:abc12345:" in key
        assert "options=" in key

    def test_cache_key_with_complex_kwarg_list(self) -> None:
        """Test cache key with complex list in kwargs (JSON serialized)."""
        key = _generate_cache_key("test_func", "abc12345", items=[{"id": 1}, {"id": 2}])

        # Should contain JSON-serialized list
        assert "result_cache:test_func:abc12345:" in key
        assert "items=" in key


class TestCachedResultDecorator:
    """Test cached_result() decorator."""

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

            # Ensure _redis_client is None initially to cover lines 36-43

            from src.cache_manager import result_cache

            result_cache._redis_client = None

            # First call - cache miss

            mock_redis.get.return_value = None

            result1 = await fetch_data("item1")

            assert result1 == {"id": "item1", "data": "test_data"}

            assert call_count == 1

            assert mock_redis.setex.called

            # Second call - cache hit

            mock_redis.get.return_value = json.dumps(
                {"id": "item1", "data": "test_data"}
            )

            result2 = await fetch_data("item1")

            assert result2 == {"id": "item1", "data": "test_data"}

            assert call_count == 1  # Function not called again

    @pytest.mark.asyncio
    async def test_close_result_cache_redis_client(self) -> None:
        """Test close_result_cache_redis_client function."""

        from redis.asyncio import Redis

        from src.cache_manager.result_cache import close_result_cache_redis_client

        mock_redis = AsyncMock(spec=Redis)

        mock_redis.aclose = AsyncMock()

        with patch("src.cache_manager.result_cache._redis_client", new=mock_redis):

            await close_result_cache_redis_client()

            mock_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_result_cache_redis_client_no_client(self) -> None:
        """Test close_result_cache_redis_client when no client is set."""

        from src.cache_manager.result_cache import close_result_cache_redis_client

        with patch("src.cache_manager.result_cache._redis_client", new=None):

            # Should not raise an error

            await close_result_cache_redis_client()

    @pytest.mark.asyncio
    async def test_decorator_cache_disabled(self) -> None:
        """Test decorator when caching is disabled."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        # Mock get_cache_config to return disabled config
        with patch("src.cache_manager.result_cache.get_cache_config") as mock_config:
            mock_config.return_value = MagicMock(enabled=False)

            result1 = await fetch_data("item1")
            result2 = await fetch_data("item1")

            # Function should be called both times (no caching)
            assert call_count == 2
            assert result1 == result2

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
            # Simulate Redis connection error
            mock_get_client.side_effect = Exception("Connection failed")

            # Should fall back to executing function without caching
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_redis_get_error(self) -> None:
        """Test decorator behavior when Redis get() fails."""
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
            mock_redis.get.side_effect = Exception("Redis get failed")

            # Should fall back to executing function
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

            # Should still return result even if caching fails
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            assert call_count == 2  # Called in try block, then again in except

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
            assert len(parts[2]) == 16  # schema hash (16 characters for collision resistance)
            assert parts[3] == "item1"  # argument

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


class TestSchemaHashInvalidation:
    """Test schema hash invalidation behavior."""

    @pytest.mark.asyncio
    async def test_function_modification_changes_cache_key(self) -> None:
        """Test that modifying function code changes cache key."""

        # Version 1 of function
        @cached_result(ttl=60, key_prefix="versioned_func")
        async def fetch_data_v1(item_id: str) -> Dict[str, str]:
            return {"id": item_id, "version": "1"}

        # Version 2 of function (different code)
        @cached_result(ttl=60, key_prefix="versioned_func")
        async def fetch_data_v2(item_id: str) -> Dict[str, str]:
            return {"id": item_id, "version": "2", "new_field": "data"}

        with patch(
            "src.cache_manager.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            # Call both versions
            await fetch_data_v1("item1")
            key_v1 = mock_redis.get.call_args_list[0][0][0]

            await fetch_data_v2("item1")
            key_v2 = mock_redis.get.call_args_list[1][0][0]

            # Cache keys should be different (different schema hashes)
            assert key_v1 != key_v2


class TestGetResultCacheRedisClient:
    """Test get_result_cache_redis_client function."""

    def setup_method(self) -> None:
        """Reset the _redis_client before each test to ensure a clean state."""
        import src.cache_manager.result_cache

        src.cache_manager.result_cache._redis_client = None

    @pytest.mark.asyncio
    async def test_initialization_and_singleton(self) -> None:
        """Test that the Redis client is initialized and is a singleton."""
        from redis.asyncio import Redis

        from src.cache_manager.config import CacheConfig
        from src.cache_manager.result_cache import get_result_cache_redis_client

        with (
            patch("src.cache_manager.result_cache.get_cache_config") as mock_get_config,
            patch(
                "src.cache_manager.result_cache.Redis.from_url"
            ) as mock_redis_from_url,
            patch("src.cache_manager.result_cache.logging") as mock_logging,
        ):

            mock_config_instance = MagicMock(spec=CacheConfig)
            mock_config_instance.redis_url = "redis://test-host:6379/1"
            mock_get_config.return_value = mock_config_instance

            mock_redis_client_instance = AsyncMock(spec=Redis)
            mock_redis_from_url.return_value = mock_redis_client_instance

            # First call - should initialize client
            client1 = await get_result_cache_redis_client()

            mock_get_config.assert_called_once()
            mock_redis_from_url.assert_called_once_with(
                "redis://test-host:6379/1", decode_responses=True
            )
            mock_logging.info.assert_called_once_with(
                "Initializing singleton Redis client for result cache: redis://test-host:6379/1"
            )
            assert client1 is mock_redis_client_instance
            from src.cache_manager.result_cache import _redis_client

            assert _redis_client is client1

            # Second call - should return existing client (singleton)
            mock_get_config.reset_mock()
            mock_redis_from_url.reset_mock()
            mock_logging.info.reset_mock()

            client2 = await get_result_cache_redis_client()

            mock_get_config.assert_not_called()
            mock_redis_from_url.assert_not_called()
            mock_logging.info.assert_not_called()
            assert client2 is client1

    @pytest.mark.asyncio
    async def test_concurrent_initialization_uses_lock(self) -> None:
        """Test that singleton initialization uses asyncio.Lock for thread safety.
        
        This test verifies the implementation uses proper async synchronization
        (asyncio.Lock with double-checked locking pattern) to prevent race conditions
        when multiple coroutines call get_result_cache_redis_client() concurrently.
        
        Expected behavior with Lock:
        - Only ONE coroutine acquires lock and initializes client
        - Others wait for lock, then see client is already initialized (double-check)
        - Result: Exactly one Redis.from_url() call, one client instance
        """
        import asyncio

        from redis.asyncio import Redis

        from src.cache_manager.config import CacheConfig
        from src.cache_manager.result_cache import get_result_cache_redis_client

        call_count = 0
        
        def counting_from_url(*args, **kwargs):
            """Count Redis.from_url calls to verify singleton behavior."""
            nonlocal call_count
            call_count += 1
            return AsyncMock(spec=Redis)

        with (
            patch("src.cache_manager.result_cache.get_cache_config") as mock_get_config,
            patch(
                "src.cache_manager.result_cache.Redis.from_url",
                side_effect=counting_from_url
            ),
        ):
            mock_config_instance = MagicMock(spec=CacheConfig)
            mock_config_instance.redis_url = "redis://test-host:6379/1"
            mock_get_config.return_value = mock_config_instance

            # Simulate concurrent access from multiple coroutines
            # With proper Lock: Only first coroutine initializes, rest wait and reuse
            results = await asyncio.gather(*[
                get_result_cache_redis_client()
                for _ in range(10)
            ])

            # REQUIREMENT: Exactly ONE initialization regardless of concurrency
            # This is guaranteed by async.Lock + double-checked locking
            assert call_count == 1, (
                f"Singleton initialization failed: Redis.from_url called {call_count} times "
                f"(expected 1). Implementation must use asyncio.Lock."
            )
            
            # All results must be identical instance
            unique_count = len(set(id(r) for r in results))
            assert unique_count == 1, f"Expected 1 unique instance, got {unique_count}"
