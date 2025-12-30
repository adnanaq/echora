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
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from http_cache.result_cache import (
    _compute_schema_hash,
    _generate_cache_key,
    cached_result,
)
from redis.exceptions import RedisError


class TestComputeSchemaHash:
    """Test _compute_schema_hash() function."""

    def test_regular_function_hash(self) -> None:
        """Test schema hash computation for regular function."""

        def test_func() -> str:
            """
            Return a constant string used for testing.

            Returns:
                result (str): The string "test".
            """
            return "test"

        hash1 = _compute_schema_hash(test_func)

        # Should be 16-character hex string (64 bits for collision resistance)
        assert len(hash1) == 16
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_same_function_same_hash(self) -> None:
        """Test that same function produces same hash."""

        def test_func() -> str:
            """
            Return a constant string used for testing.

            Returns:
                result (str): The string "test".
            """
            return "test"

        hash1 = _compute_schema_hash(test_func)
        hash2 = _compute_schema_hash(test_func)

        assert hash1 == hash2

    def test_different_functions_different_hash(self) -> None:
        """Test that different functions produce different hashes."""

        def func1() -> str:
            """
            Return the fixed string "func1".

            Returns:
                str: The literal string "func1".
            """
            return "func1"

        def func2() -> str:
            """
            Return the literal string 'func2'.

            Returns:
                result (str): The literal string "func2".
            """
            return "func2"

        hash1 = _compute_schema_hash(func1)
        hash2 = _compute_schema_hash(func2)

        assert hash1 != hash2

    def test_function_code_change_changes_hash(self) -> None:
        """Test that changing function code changes hash."""

        def func_v1() -> str:
            """
            Return a fixed identifier for this function implementation version.

            Returns:
                str: The literal string "version 1" identifying this version.
            """
            return "version 1"

        hash1 = _compute_schema_hash(func_v1)

        def func_v2() -> str:
            """
            Return a literal indicating the function's version.

            Returns:
                version (str): The string "version 2".
            """
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
        expected_hash = hashlib.sha256(b"len").hexdigest()[:16]
        assert hash_result == expected_hash

    def test_function_with_multiline_code(self) -> None:
        """Test hash computation for functions with multiline code."""

        def complex_func(x: int, y: int) -> int:
            """
            Compute the sum of x and y, doubling the result when the sum is greater than 10.

            Returns:
                The sum of x and y; if the sum is greater than 10, returns that sum multiplied by 2.
            """
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
            """
            Create a simple decorator that forwards calls to the decorated function.

            Returns:
                A wrapper function that calls the original `func` with the same positional and
                keyword arguments and returns its result.
            """

            def wrapper(*args, **kwargs):  # type: ignore
                return func(*args, **kwargs)

            return wrapper

        @decorator
        def decorated_func() -> str:
            """
            Provide the constant string "decorated".

            Returns:
                str: The string "decorated".
            """
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
        """Test that very long cache keys are hashed instead of including full argument."""
        from http_cache.config import get_cache_config

        # Create a key longer than 200 characters
        long_arg = "x" * 200

        key = _generate_cache_key("test_func", "abc12345", long_arg)

        # Should contain prefix and schema hash
        assert "result_cache:test_func:abc12345:" in key
        # Should NOT contain the full long argument (proves it was hashed)
        assert long_arg not in key
        # Should respect configured max cache key length
        assert len(key) <= get_cache_config().max_cache_key_length

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

    def test_cache_key_with_non_json_serializable_arg(self) -> None:
        """Test cache key with non-JSON-serializable argument falls back to repr()."""
        # datetime is not JSON-serializable by default
        dt = datetime(2025, 1, 20, 12, 30, 45)

        key = _generate_cache_key("test_func", "abc12345", dt)

        # Should contain repr() representation instead of crashing
        assert "result_cache:test_func:abc12345:" in key
        assert "2025" in key  # repr of datetime contains the year

    def test_cache_key_with_non_json_serializable_kwarg(self) -> None:
        """Test cache key with non-JSON-serializable kwarg falls back to repr()."""
        dt = datetime(2025, 1, 20, 12, 30, 45)

        key = _generate_cache_key("test_func", "abc12345", timestamp=dt)

        # Should contain repr() representation instead of crashing
        assert "result_cache:test_func:abc12345:" in key
        assert "timestamp=" in key
        assert "2025" in key

    def test_cache_key_with_custom_class_arg(self) -> None:
        """Test cache key with custom class instance falls back to repr()."""

        class CustomObject:
            def __init__(self, value: str):
                """
                Initialize the instance with a string value.

                Parameters:
                    value (str): The string assigned to the instance's `value` attribute.
                """
                self.value = value

            def __repr__(self) -> str:
                """
                Return the canonical string representation of the CustomObject for debugging.

                Returns:
                    repr_str (str): A string in the form "CustomObject(value=...)" where the contained value is represented using its own `repr`.
                """
                return f"CustomObject(value={self.value!r})"

        obj = CustomObject("test123")
        key = _generate_cache_key("test_func", "abc12345", obj)

        # Should contain repr() representation
        assert "result_cache:test_func:abc12345:" in key
        assert "CustomObject" in key or "test123" in key


class TestCachedResultDecorator:
    """Test cached_result() decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic_usage(self) -> None:
        """Test basic decorator usage with cache miss and hit."""

        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, Any]:
            """Fetch test data and increment call counter."""

            nonlocal call_count

            call_count += 1

            return {"id": item_id, "data": "test_data"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()

            mock_get_client.return_value = mock_redis

            # Ensure _redis_client is None initially to cover lines 36-43

            from http_cache import result_cache

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

        from http_cache.result_cache import close_result_cache_redis_client
        from redis.asyncio import Redis

        mock_redis = AsyncMock(spec=Redis)

        mock_redis.aclose = AsyncMock()

        with patch("http_cache.result_cache._redis_client", new=mock_redis):
            await close_result_cache_redis_client()

            mock_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_result_cache_redis_client_no_client(self) -> None:
        """Test close_result_cache_redis_client when no client is set."""

        from http_cache.result_cache import close_result_cache_redis_client

        with patch("http_cache.result_cache._redis_client", new=None):
            # Should not raise an error

            await close_result_cache_redis_client()

    @pytest.mark.asyncio
    async def test_decorator_cache_disabled(self) -> None:
        """Test decorator when caching is disabled."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, Any]:
            """
            Fetch test data for the given item identifier.

            Parameters:
                item_id (str): Identifier of the item to fetch.

            Returns:
                dict: A dictionary with "id" set to the provided identifier and "data" containing a test payload.
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        # Mock get_cache_config to return disabled config
        with patch("http_cache.result_cache.get_cache_config") as mock_config:
            mock_config.return_value = MagicMock(enabled=False)

            result1 = await fetch_data("item1")
            result2 = await fetch_data("item1")

            # Function should be called both times (no caching)
            assert call_count == 2
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_decorator_with_default_ttl(self) -> None:
        """
        Verifies the cached_result decorator applies the default TTL when no `ttl` is specified.

        Patches the Redis client to simulate a cache miss and asserts that the decorator calls `setex` with the default TTL of 86400 seconds (24 hours).
        """

        @cached_result(key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Return a mapping containing the provided item identifier under the "id" key.

            Returns:
                dict: A dictionary with a single entry `"id": item_id`.
            """
            return {"id": item_id}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            await fetch_data("item1")

            # Should use default 24h TTL (86400 seconds)
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            # Access TTL more resiliently - check kwargs first, fall back to positional
            ttl = (
                call_args.kwargs.get("time") if call_args.kwargs else call_args.args[1]
            )
            assert ttl == 86400  # Default TTL

    @pytest.mark.asyncio
    async def test_decorator_with_custom_ttl(self) -> None:
        """Test decorator with custom TTL."""

        @cached_result(ttl=3600, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Return a mapping containing the provided item identifier under the "id" key.

            Returns:
                dict: A dictionary with a single entry `"id": item_id`.
            """
            return {"id": item_id}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            await fetch_data("item1")

            # Should use custom TTL
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            # Access TTL more resiliently - check kwargs first, fall back to positional
            ttl = (
                call_args.kwargs.get("time") if call_args.kwargs else call_args.args[1]
            )
            assert ttl == 3600

    @pytest.mark.asyncio
    async def test_decorator_uses_function_name_as_default_prefix(self) -> None:
        """Test that decorator uses function name as default key prefix."""

        @cached_result(ttl=60)
        async def my_custom_function(item_id: str) -> dict[str, str]:
            """
            Create a minimal mapping containing the provided item identifier.

            Returns:
                dict: A dictionary with the key "id" whose value is the given `item_id`.
            """
            return {"id": item_id}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
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
        async def fetch_data(item_id: str) -> dict[str, str] | None:  # noqa: ARG001
            """
            Simulates fetching data for the given item while incrementing a call counter.

            Increments the enclosing scope's `call_count` nonlocal variable to record an invocation and always returns None.

            Returns:
                None
            """
            nonlocal call_count
            call_count += 1
            return None

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
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
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Fetches mock data for a given item identifier and records the invocation.

            Parameters:
                item_id (str): The identifier of the item to fetch.

            Returns:
                dict: A mapping with keys `"id"` set to the provided `item_id` and `"data"` set to the string `"test"`.
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            # Simulate Redis connection error
            mock_get_client.side_effect = RedisError("Connection failed")

            # Should fall back to executing function without caching
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_redis_get_error(self) -> None:
        """Test decorator behavior when Redis get() fails."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Fetches mock data for a given item identifier and records the invocation.

            Parameters:
                item_id (str): The identifier of the item to fetch.

            Returns:
                dict: A mapping with keys `"id"` set to the provided `item_id` and `"data"` set to the string `"test"`.
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.side_effect = RedisError("Redis get failed")

            # Should fall back to executing function
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_redis_set_error(self) -> None:
        """Test decorator behavior when Redis setex() fails.

        CRITICAL: Function should be called exactly ONCE even if cache write fails.
        Cache write errors should NOT trigger re-execution of expensive operations.
        """
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Fetches mock data for a given item identifier and records the invocation.

            Parameters:
                item_id (str): The identifier of the item to fetch.

            Returns:
                dict: A mapping with keys `"id"` set to the provided `item_id` and `"data"` set to the string `"test"`.
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None
            mock_redis.setex.side_effect = RedisError("Redis setex failed")

            # Should still return result even if caching fails
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            # FIXED: Should be called only ONCE (in try block)
            # Cache write failure should NOT cause re-execution
            assert call_count == 1, (
                f"Function called {call_count} times. "
                f"Expected 1 call even with cache write error."
            )

    @pytest.mark.asyncio
    async def test_decorator_json_dumps_error(self) -> None:
        """Test decorator behavior when json.dumps() fails.

        CRITICAL: Function should be called exactly ONCE even if serialization fails.
        Serialization errors should NOT trigger re-execution of expensive operations.
        """
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Fetches mock data for a given item identifier and records the invocation.

            Parameters:
                item_id (str): The identifier of the item to fetch.

            Returns:
                dict: A mapping with keys `"id"` set to the provided `item_id` and `"data"` set to the string `"test"`.
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with (
            patch(
                "http_cache.result_cache.get_result_cache_redis_client"
            ) as mock_get_client,
            patch("http_cache.result_cache.json.dumps") as mock_dumps,
        ):
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis
            mock_redis.get.return_value = None

            # Simulate json.dumps failure (e.g., non-serializable object)
            mock_dumps.side_effect = TypeError("Object is not JSON serializable")

            # Should still return result even if serialization fails
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            # FIXED: Should be called only ONCE (in try block)
            # Serialization failure should NOT cause re-execution
            assert call_count == 1, (
                f"Function called {call_count} times. "
                f"Expected 1 call even with json.dumps error."
            )

    @pytest.mark.asyncio
    async def test_decorator_corrupted_cache_data(self) -> None:
        """Test decorator behavior when cached data is corrupted (invalid JSON).

        CRITICAL: Corrupted cache should be treated as cache miss, not crash.
        Function should be called once and return correct result.
        """
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Fetches mock data for a given item identifier and records the invocation.

            Parameters:
                item_id (str): The identifier of the item to fetch.

            Returns:
                dict: A mapping with keys `"id"` set to the provided `item_id` and `"data"` set to the string `"test"`.
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": "test"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis

            # Return corrupted/invalid JSON from cache
            mock_redis.get.return_value = "{invalid json data"

            # Should treat as cache miss and execute function
            result = await fetch_data("item1")

            assert result == {"id": "item1", "data": "test"}
            # Should call function once (cache miss due to corruption)
            assert call_count == 1, (
                f"Function called {call_count} times. "
                f"Expected 1 call when cache data is corrupted."
            )

    @pytest.mark.asyncio
    async def test_decorator_different_args_different_cache_entries(self) -> None:
        """Test that different arguments create different cache entries."""
        call_count = 0

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Return a simple data record for the given item identifier.

            Parameters:
                item_id (str): Identifier of the item.

            Returns:
                Dict[str, str]: A dictionary containing 'id' set to the provided `item_id` and 'data' set to "data_{item_id}".
            """
            nonlocal call_count
            call_count += 1
            return {"id": item_id, "data": f"data_{item_id}"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
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
        async def fetch_data(item_id: str) -> dict[str, str]:
            """
            Return a mapping containing the provided item identifier under the "id" key.

            Returns:
                dict: A dictionary with a single entry `"id": item_id`.
            """
            return {"id": item_id}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
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
            assert (
                len(parts[2]) == 16
            )  # schema hash (16 characters for collision resistance)
            assert parts[3] == "item1"  # argument

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function name and docstring."""

        @cached_result(ttl=60, key_prefix="test")
        async def my_function(item_id: str) -> dict[str, str]:
            """My function docstring."""
            return {"id": item_id}

        # functools.wraps should preserve metadata
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My function docstring."

    @pytest.mark.asyncio
    async def test_decorator_with_complex_return_type(self) -> None:
        """Test decorator with complex nested return types."""

        @cached_result(ttl=60, key_prefix="test")
        async def fetch_complex_data(item_id: str) -> dict[str, Any]:
            """
            Return a structured dictionary representing complex data for the given item ID.

            Parameters:
                item_id (str): Identifier of the item to fetch.

            Returns:
                data (dict): Dictionary with keys:
                    - "id" (str): the provided item_id
                    - "nested" (dict): contains "key" (str) and "list" (list of ints)
                    - "array" (list): list of strings
                    - "number" (int): integer value
                    - "float" (float): floating-point value
                    - "bool" (bool): boolean value
                    - "null" (None): null value
            """
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
            "http_cache.result_cache.get_result_cache_redis_client"
        ) as mock_get_client:
            mock_redis = AsyncMock()
            mock_get_client.return_value = mock_redis

            # Cache miss - store complex data
            mock_redis.get.return_value = None
            result1 = await fetch_complex_data("item1")

            # Verify complex data serialized
            assert mock_redis.setex.called
            call_args = mock_redis.setex.call_args
            # Access payload more resiliently - check kwargs first, fall back to positional
            stored_data = (
                call_args.kwargs.get("value") if call_args.kwargs else call_args.args[2]
            )
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
        async def fetch_data_v1(item_id: str) -> dict[str, str]:
            """
            Return a simple data dictionary for the given item identifier indicating version 1.

            Parameters:
                item_id (str): Identifier of the item to fetch.

            Returns:
                Dict[str, str]: Dictionary with 'id' set to the provided identifier and 'version' set to "1".
            """
            return {"id": item_id, "version": "1"}

        # Version 2 of function (different code)
        @cached_result(ttl=60, key_prefix="versioned_func")
        async def fetch_data_v2(item_id: str) -> dict[str, str]:
            """
            Builds a versioned data payload for the given item identifier.

            Returns:
                dict: A mapping containing 'id' (the provided item_id), 'version' set to "2", and 'new_field' set to "data".
            """
            return {"id": item_id, "version": "2", "new_field": "data"}

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client"
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

    @pytest.mark.asyncio
    async def test_initialization_and_singleton(self) -> None:
        """Test that the Redis client is initialized and is a singleton."""
        from http_cache.config import CacheConfig
        from http_cache.result_cache import get_result_cache_redis_client
        from redis.asyncio import Redis

        with (
            patch("http_cache.result_cache.get_cache_config") as mock_get_config,
            patch("http_cache.result_cache.Redis.from_url") as mock_redis_from_url,
            patch("http_cache.result_cache.logging") as mock_logging,
        ):
            mock_config_instance = MagicMock(spec=CacheConfig)
            mock_config_instance.redis_url = "redis://test-host:6379/1"
            mock_config_instance.redis_max_connections = 100
            mock_config_instance.redis_socket_keepalive = True
            mock_config_instance.redis_socket_connect_timeout = 5
            mock_config_instance.redis_socket_timeout = 10
            mock_config_instance.redis_retry_on_timeout = True
            mock_config_instance.redis_health_check_interval = 30
            mock_get_config.return_value = mock_config_instance

            mock_redis_client_instance = AsyncMock(spec=Redis)
            mock_redis_from_url.return_value = mock_redis_client_instance

            # First call - should initialize client
            client1 = await get_result_cache_redis_client()

            mock_get_config.assert_called_once()
            mock_redis_from_url.assert_called_once_with(
                "redis://test-host:6379/1",
                decode_responses=True,
                max_connections=100,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            # Check log message contains the URL and connection pool info
            assert mock_logging.info.call_count == 1
            log_call_args = mock_logging.info.call_args[0][0]
            assert "redis://test-host:6379/1" in log_call_args
            assert "max_connections=100" in log_call_args
            assert client1 is mock_redis_client_instance
            from http_cache.result_cache import _redis_client

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

        from http_cache.config import CacheConfig
        from http_cache.result_cache import get_result_cache_redis_client
        from redis.asyncio import Redis

        call_count = 0

        def counting_from_url(*_args, **_kwargs):
            """
            Count Redis.from_url calls to verify singleton behavior.

            Increments the enclosing `call_count` counter each time it is invoked and returns
            an AsyncMock that acts like a Redis client.

            Returns:
                AsyncMock: An AsyncMock instance with the Redis spec that simulates a Redis client.
            """
            nonlocal call_count
            call_count += 1
            return AsyncMock(spec=Redis)

        with (
            patch("http_cache.result_cache.get_cache_config") as mock_get_config,
            patch(
                "http_cache.result_cache.Redis.from_url", side_effect=counting_from_url
            ),
        ):
            mock_config_instance = MagicMock(spec=CacheConfig)
            mock_config_instance.redis_url = "redis://test-host:6379/1"
            mock_config_instance.redis_max_connections = 100
            mock_config_instance.redis_socket_keepalive = True
            mock_config_instance.redis_socket_connect_timeout = 5
            mock_config_instance.redis_socket_timeout = 10
            mock_config_instance.redis_retry_on_timeout = True
            mock_config_instance.redis_health_check_interval = 30
            mock_get_config.return_value = mock_config_instance

            # Simulate concurrent access from multiple coroutines
            # With proper Lock: Only first coroutine initializes, rest wait and reuse
            results = await asyncio.gather(
                *[get_result_cache_redis_client() for _ in range(10)]
            )

            # REQUIREMENT: Exactly ONE initialization regardless of concurrency
            # This is guaranteed by async.Lock + double-checked locking
            assert call_count == 1, (
                f"Singleton initialization failed: Redis.from_url called {call_count} times "
                f"(expected 1). Implementation must use asyncio.Lock."
            )

            # All results must be identical instance
            unique_count = len(set(id(r) for r in results))
            assert unique_count == 1, f"Expected 1 unique instance, got {unique_count}"

    @pytest.mark.asyncio
    async def test_no_race_condition_with_concurrent_close(self) -> None:
        """Test that get_result_cache_redis_client never returns None during concurrent close.

        Race condition scenario:
        1. Coroutine A checks if client is None (passes, client exists)
        2. Coroutine B closes client, sets to None
        3. Coroutine A returns _redis_client (now None!)

        This test verifies the fix: holding lock until return prevents this race.
        """
        import asyncio

        from http_cache.config import CacheConfig
        from http_cache.result_cache import (
            close_result_cache_redis_client,
            get_result_cache_redis_client,
        )
        from redis.asyncio import Redis

        with (
            patch("http_cache.result_cache.get_cache_config") as mock_get_config,
            patch("http_cache.result_cache.Redis.from_url") as mock_from_url,
        ):
            mock_config_instance = MagicMock(spec=CacheConfig)
            mock_config_instance.redis_url = "redis://localhost:6379/0"
            mock_config_instance.redis_max_connections = 100
            mock_config_instance.redis_socket_keepalive = True
            mock_config_instance.redis_socket_connect_timeout = 5
            mock_config_instance.redis_socket_timeout = 10
            mock_config_instance.redis_retry_on_timeout = True
            mock_config_instance.redis_health_check_interval = 30
            mock_get_config.return_value = mock_config_instance

            # Create mock clients
            mock_client_1 = AsyncMock(spec=Redis)
            mock_client_2 = AsyncMock(spec=Redis)
            mock_from_url.side_effect = [mock_client_1, mock_client_2]

            # Initialize client first
            initial_client = await get_result_cache_redis_client()
            assert initial_client is not None
            assert initial_client == mock_client_1

            # Track results from concurrent operations
            get_results = []
            none_returned = False

            # Use events for coordination instead of sleep-based timing
            getter_started = asyncio.Event()
            closer_can_start = asyncio.Event()

            async def concurrent_getter():
                """Try to get client multiple times during close."""
                nonlocal none_returned
                getter_started.set()  # Signal that getter has started
                await closer_can_start.wait()  # Wait for signal to ensure overlap
                for _ in range(10):
                    client = await get_result_cache_redis_client()
                    get_results.append(client)
                    if client is None:
                        none_returned = True
                    await asyncio.sleep(0.001)  # Small sleep to yield control

            async def concurrent_closer():
                """
                Waits for getter to start, then closes the global Redis result cache client.

                Used in tests to simulate closing the cache client while concurrent get operations are in progress.
                """
                await getter_started.wait()  # Wait for getter to start
                closer_can_start.set()  # Signal getter can proceed
                await asyncio.sleep(
                    0.001
                )  # Brief delay to ensure getter is mid-execution
                await close_result_cache_redis_client()

            # Run concurrently to trigger potential race
            await asyncio.gather(concurrent_getter(), concurrent_closer())

            # CRITICAL: After fix, should NEVER return None
            assert not none_returned, (
                "Race condition detected: get_result_cache_redis_client() "
                "returned None during concurrent close operation"
            )

            # All results must be valid Redis clients (not None)
            for i, result in enumerate(get_results):
                assert result is not None, (
                    f"Result at index {i} was None - race condition exists"
                )


class TestMaxCacheKeyLength:
    """Test max_cache_key_length configuration parameter."""

    def test_default_max_cache_key_length(self) -> None:
        """Test that default max_cache_key_length is 200."""
        from http_cache.config import get_cache_config

        config = get_cache_config()
        assert config.max_cache_key_length == 200

    def test_custom_max_cache_key_length_via_config(self) -> None:
        """Test that custom max_cache_key_length can be set via config."""
        from http_cache.config import CacheConfig

        config = CacheConfig(max_cache_key_length=500)
        assert config.max_cache_key_length == 500
