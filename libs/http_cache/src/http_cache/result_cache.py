"""
Result-level caching for crawler functions.

Since crawlers use browser automation (crawl4ai/Playwright) rather than HTTP libraries,
we cache the final extracted results instead of HTTP responses.

Schema hashing: Cache keys include a hash of the function's source code.
When code changes (CSS selectors, extraction logic, etc.), the hash changes,
automatically invalidating old cache entries.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import get_cache_config

# Type variables for preserving function signatures
P = ParamSpec("P")
R = TypeVar("R")

# --- Singleton Redis Client for @cached_result ---

_redis_lock = asyncio.Lock()
_redis_client: Redis | None = None


async def get_result_cache_redis_client() -> Redis:
    """
    Return the singleton async Redis client used for result-level caching.

    This function initializes the module-level Redis client once and returns the same instance on subsequent calls. Initialization is guarded by an internal async lock so that concurrent callers cannot race with client closure; callers can rely on a single shared client for the process lifetime until closed.

    Returns:
        Redis: The initialized singleton Redis client for result caching.

    Raises:
        RuntimeError: If client initialization fails and no Redis instance could be created.
    """
    global _redis_client
    async with _redis_lock:
        # Hold lock until return to ensure atomic check-and-return
        # Prevents race: close() can't set to None while we hold the lock
        if _redis_client is None:
            config = get_cache_config()
            redis_url = config.redis_url or "redis://localhost:6379/0"
            logging.info(
                f"Initializing singleton Redis client for result cache: {redis_url} "
                f"(max_connections={config.redis_max_connections})"
            )
            # Configure connection pool for multi-agent concurrency and reliability
            _redis_client = Redis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=config.redis_max_connections,
                socket_keepalive=config.redis_socket_keepalive,
                socket_connect_timeout=config.redis_socket_connect_timeout,
                socket_timeout=config.redis_socket_timeout,
                retry_on_timeout=config.redis_retry_on_timeout,
                health_check_interval=config.redis_health_check_interval,
            )
        # After initialization, client must be non-None
        if _redis_client is None:
            raise RuntimeError("Failed to initialize Redis client for result cache")
        return _redis_client


async def close_result_cache_redis_client() -> None:
    """
    Close the module's singleton Redis client used for result caching.

    If a client exists, acquires the module's global lock, closes the client asynchronously, and clears the singleton reference. Safe to call when no client is initialized (no-op).
    """
    global _redis_client
    async with _redis_lock:
        if _redis_client:
            logging.info("Closing singleton Redis client for result cache.")
            await _redis_client.aclose()
            _redis_client = None


# --- End Singleton ---


# Type variable for generic function return type
T = TypeVar("T")


def _compute_schema_hash(func: Callable[..., Any]) -> str:
    """
    Compute a hash of the function's source code.

    This hash changes whenever the function's implementation changes,
    automatically invalidating cached results when code is modified.

    Args:
        func: Function to hash

    Returns:
        16-character hexadecimal hash of the function's source code (64 bits)
    """
    try:
        # Get the source code of the function
        source = inspect.getsource(func)
        # Generate SHA-256 hash and take first 16 characters for 64-bit collision resistance
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    except (OSError, TypeError):
        # If we can't get source (built-in, lambda, etc.), use function name
        return hashlib.sha256(func.__name__.encode()).hexdigest()[:16]


def _generate_cache_key(
    prefix: str, schema_hash: str, *args: Any, **kwargs: Any
) -> str:
    """
    Create a stable Redis cache key for a function call that includes the provided prefix and schema hash.

    Serializes positional and keyword arguments deterministically (keyword args sorted by key). Basic types (str, int, float, bool, None) are converted to strings; other values are JSON-serialized with sort_keys=True when possible and fall back to repr() on failure. If the assembled key exceeds the configured max cache key length, the function returns a hashed form to enforce the length limit.

    Parameters:
        prefix (str): Cache key prefix, typically the function name.
        schema_hash (str): Short hash representing the function's source/schema.
        *args: Positional arguments to include in the key.
        **kwargs: Keyword arguments to include in the key.

    Returns:
        str: A Redis key in one of two forms:
             - "result_cache:{prefix}:{schema_hash}:{...parts...}" when the assembled key is within length limits.
             - "result_cache:{prefix}:{schema_hash}:{sha256_hex}" when the assembled key exceeds the configured max length.
    """
    # Get max key length from config
    config = get_cache_config()
    max_key_length = config.max_cache_key_length

    # Serialize arguments to create stable key
    key_parts = [prefix, schema_hash]  # Include schema hash

    # Add positional args
    for arg in args:
        if isinstance(arg, str | int | float | bool):
            key_parts.append(str(arg))
        else:
            # For complex types, use JSON serialization with fallback to repr()
            try:
                key_parts.append(json.dumps(arg, sort_keys=True))
            except TypeError:
                # Fall back to repr() for non-JSON-serializable objects
                key_parts.append(repr(arg))

    # Add keyword args (sorted for stability)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, str | int | float | bool | type(None)):
            key_parts.append(f"{k}={v}")
        else:
            # For complex types, use JSON serialization with fallback to repr()
            try:
                key_parts.append(f"{k}={json.dumps(v, sort_keys=True)}")
            except TypeError:
                # Fall back to repr() for non-JSON-serializable objects
                key_parts.append(f"{k}={repr(v)}")

    # Hash the combined key if it exceeds threshold
    key_string = ":".join(key_parts)
    if len(key_string) > max_key_length:
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"result_cache:{prefix}:{schema_hash}:{key_hash}"

    return f"result_cache:{key_string}"


def cached_result(
    ttl: int | None = None,
    key_prefix: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator to cache async function results in Redis with automatic schema invalidation.

    The cache key includes a hash of the function's source code. When you modify
    the function (change CSS selectors, extraction logic, etc.), the hash changes
    and old cache entries are automatically invalidated.

    Usage:
        @cached_result(ttl=86400, key_prefix="animeplanet_anime")
        async def fetch_anime(slug: str) -> Optional[Dict[str, Any]]:
            # Expensive crawler operation
            return data

    Args:
        ttl: Time-to-live in seconds (None = use fixed 24h default, independent of CacheConfig)
        key_prefix: Optional custom key prefix (default: function name)

    Returns:
        Decorated function with caching

    Note:
        The default TTL is a fixed 24 hours (86400 seconds), independent of CacheConfig.
        This is intentional as result caching serves a different purpose than HTTP caching.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        # Compute schema hash once when decorator is applied
        """
        Wraps an async function to cache its return value in Redis using a schema-hash-based key.

        The returned wrapper will return a cached JSON-deserialized value when available; on a cache miss it will call the original function, store the JSON-serializable result in Redis, and return it. Caching is skipped and the original function is invoked if caching is disabled or Redis is not configured; Redis read/write failures are treated as best-effort and fall back to calling the original function. The wrapper never caches `None`. Cache keys include a hash of the wrapped function's source to invalidate entries when the function implementation changes. When `ttl` is not provided, a default TTL of 86400 seconds (24 hours) is used.

        Parameters:
            func (Callable): The async function to wrap.

        Returns:
            Callable: An async wrapper that returns the cached or newly computed result.
        """
        schema_hash = _compute_schema_hash(func)

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get cache config
            """
            Cache and return the wrapped function's result in Redis using a schema-based key.

            Attempts to read a JSON-serialized cached value keyed by a prefix and a hash of the wrapped function's source; on cache hit the deserialized value is returned. On cache miss the wrapped function is executed exactly once, its non-None result is JSON-serialized and stored in Redis with the specified TTL (default 86400 seconds). Cache read/write errors or JSON decode/encode failures are treated as best-effort and do not change the function's behavior. `None` results are returned but not cached.

            Returns:
                The wrapped function's result; a cached value when available, otherwise the freshly computed result.
            """
            config = get_cache_config()

            # Skip caching if disabled
            if not config.enabled or config.storage_type != "redis":
                return await func(*args, **kwargs)

            # Generate cache key with schema hash
            prefix = key_prefix or func.__name__
            cache_key = _generate_cache_key(prefix, schema_hash, *args, **kwargs)

            try:
                # Get singleton Redis client and attempt cache read
                redis_client = await get_result_cache_redis_client()
                cached_data = await redis_client.get(cache_key)
            except RedisError as e:
                # On cache-read errors, fall back to direct call
                logging.warning(f"Cache read error in {func.__name__}: {e}")
                return await func(*args, **kwargs)

            if cached_data:
                try:
                    # Deserialize cached data
                    return cast(R, json.loads(cached_data))
                except json.JSONDecodeError as e:
                    # Corrupted cache entry - treat as cache miss
                    logging.warning(
                        f"Cache decode error in {func.__name__} for key {cache_key}: {e}"
                    )
                    # Fall through to execute function (cache miss)

            # Cache miss - execute function exactly once
            result = await func(*args, **kwargs)

            if result is None:
                return result

            try:
                # Best-effort cache write; failures should not re-call func
                serialized = json.dumps(result, ensure_ascii=False)
                # Fixed 24h default TTL (independent of CacheConfig, which is for HTTP caching)
                cache_ttl = ttl if ttl is not None else 86400
                await redis_client.setex(cache_key, cache_ttl, serialized)
            except (RedisError, TypeError) as e:
                # RedisError: Redis connection/operation failures
                # TypeError: JSON serialization fails for non-serializable objects
                logging.warning(f"Cache write error in {func.__name__}: {e}")

            return result

        return wrapper

    return decorator
