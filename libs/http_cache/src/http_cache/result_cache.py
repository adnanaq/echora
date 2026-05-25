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
from typing import Any, Generic, ParamSpec, TypeVar, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import get_cache_config
from .exceptions import RedisInitializationError
from .utils import _mask_url_credentials

P = ParamSpec("P")
R = TypeVar("R")


# --- Singleton Redis Client for @cached_result ---

_redis_lock = asyncio.Lock()
_redis_client: Redis | None = None


async def get_result_cache_redis_client() -> Redis:
    """Return the singleton async Redis client used for result-level caching.

    Initializes the client on first call; subsequent calls return the same
    instance. Guarded by an async lock so concurrent callers cannot race
    with client closure.

    Returns:
        The initialized singleton Redis client.

    Raises:
        RedisInitializationError: If the client cannot be created.
    """
    global _redis_client
    async with _redis_lock:
        # Hold lock until return to ensure atomic check-and-return
        # Prevents race: close() can't set to None while we hold the lock
        if _redis_client is None:
            config = get_cache_config()
            redis_url = config.redis_url or "redis://localhost:6379/0"
            logging.info(
                f"Initializing singleton Redis client for result cache: {_mask_url_credentials(redis_url)} "
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
            raise RedisInitializationError()
        return _redis_client


async def close_result_cache_redis_client() -> None:
    """Close the singleton Redis client used for result caching.

    Acquires the global lock, closes the client asynchronously, and clears
    the singleton reference. Safe to call when no client is initialized (no-op).
    """
    global _redis_client
    async with _redis_lock:
        if _redis_client:
            logging.info("Closing singleton Redis client for result cache.")
            await _redis_client.aclose()
            _redis_client = None


# --- End Singleton ---


def _compute_schema_hash(
    func: Callable[..., Any], dependencies: list[Callable[..., Any]] | None = None
) -> str:
    """Compute a 16-character hex hash of a function's source code.

    When the function's implementation changes, the hash changes, automatically
    invalidating cached results. If ``dependencies`` are provided, their source
    is appended before hashing so changes to any helper also invalidate the cache.
    For callables whose source cannot be retrieved (built-ins, C extensions,
    lambdas), ``__name__`` is used as a stable fallback.

    Args:
        func: Function to hash.
        dependencies: Callables whose source should also be included in the hash.

    Returns:
        A 16-character hexadecimal hash string (64-bit SHA-256 prefix).
    """
    try:
        source = inspect.getsource(func)
        if dependencies:
            for dep in dependencies:
                try:
                    source += inspect.getsource(dep)
                except (OSError, TypeError):
                    source += getattr(dep, "__name__", repr(dep))
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    except (OSError, TypeError):
        # If we can't get source (built-in, lambda, etc.), use function name
        return hashlib.sha256(
            getattr(func, "__name__", repr(func)).encode()
        ).hexdigest()[:16]


def _generate_cache_key(
    prefix: str, schema_hash: str, *args: Any, **kwargs: Any
) -> str:
    """Create a stable Redis cache key for a function call.

    Serializes arguments deterministically: primitives (str, int, float, bool,
    None) as strings; complex types as JSON with sort_keys, falling back to
    repr(). Keyword args are sorted by key. If the assembled key exceeds the
    configured max length it is replaced with a SHA-256 hash.

    Args:
        prefix: Cache key prefix, typically the function name.
        schema_hash: Short hash representing the function's source/schema.
        *args: Positional arguments to include in the key.
        **kwargs: Keyword arguments to include in the key.

    Returns:
        A Redis key of the form ``result_cache:{prefix}:{schema_hash}:{parts}``
        or ``result_cache:{prefix}:{schema_hash}:{sha256_hex}`` when the key
        exceeds the configured max length.
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


def _normalize_batch_args(
    args_list: list[tuple[Any, ...]] | list[Any],
    kwargs_list: list[dict[str, Any]] | None,
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    """Normalize a batch args list and default kwargs_list.

    Wraps bare non-tuple items in a single-element tuple so every entry is a
    ``tuple[Any, ...]``. Defaults ``kwargs_list`` to a list of empty dicts when
    None is passed.

    Args:
        args_list: Positional-args tuples, or bare values (auto-wrapped).
        kwargs_list: Per-call keyword-arg dicts aligned to ``args_list``, or
            None to use empty dicts for all calls.

    Returns:
        A ``(normalized_args, kwargs_list)`` pair ready for cache-key generation.

    Raises:
        ValueError: If ``kwargs_list`` is provided but its length differs from
            ``args_list``.
    """
    normalized = [item if isinstance(item, tuple) else (item,) for item in args_list]
    if kwargs_list is None:
        return normalized, [{} for _ in normalized]
    if len(kwargs_list) != len(normalized):
        raise ValueError("kwargs_list length must match args_list length")
    return normalized, kwargs_list


class _CachedFunction(Generic[P, R]):
    """Async callable produced by ``@cached_result`` with batch cache helpers.

    Wraps the original function and exposes ``cache_batch_get`` / ``cache_batch_set``
    as typed methods so callers do not need dynamic attribute access.
    """

    def __init__(
        self,
        func: Callable[P, Awaitable[R]],
        schema_hash: str,
        prefix: str,
        ttl: int | None,
    ) -> None:
        """Initialise the cached wrapper.

        Args:
            func: The async function being wrapped.
            schema_hash: Pre-computed source hash for cache-key generation.
            prefix: Cache-key prefix (function name or custom override).
            ttl: Time-to-live in seconds; None defaults to 86400 (24 h).
        """
        self._func = func
        self._func_name: str = getattr(func, "__name__", repr(func))
        self._schema_hash = schema_hash
        self._prefix = prefix
        self._ttl = ttl
        functools.update_wrapper(self, func)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Return the cached result, or call the function and cache its return value.

        On a cache hit, returns the deserialized value without calling the
        original function. On a miss, executes the function once, stores the
        non-None result with SETEX, and returns it. Redis and JSON errors are
        best-effort and fall back to calling the original function. ``None``
        results are returned but not cached.

        Returns:
            The cached or freshly computed result of the wrapped function.
        """
        config = get_cache_config()

        if not config.cache_enabled or config.storage_type != "redis":
            return await self._func(*args, **kwargs)

        cache_key = _generate_cache_key(
            self._prefix, self._schema_hash, *args, **kwargs
        )

        try:
            redis_client = await get_result_cache_redis_client()
            cached_data = await redis_client.get(cache_key)
        except RedisError as e:
            logging.warning(f"Cache read error in {self._func_name}: {e}")
            return await self._func(*args, **kwargs)

        if cached_data:
            try:
                return cast(R, json.loads(cached_data))
            except json.JSONDecodeError as e:
                # Corrupted cache entry — treat as miss and fall through
                logging.warning(
                    f"Cache decode error in {self._func_name} for key {cache_key}: {e}"
                )

        result = await self._func(*args, **kwargs)

        if result is None:
            return result

        try:
            # Best-effort write; failures must not re-invoke the function
            serialized = json.dumps(result, ensure_ascii=False)
            cache_ttl = self._ttl if self._ttl is not None else 86400
            await redis_client.setex(cache_key, cache_ttl, serialized)
        except (RedisError, TypeError) as e:
            # RedisError: Redis connection/operation failures
            # TypeError: JSON serialization fails for non-serializable objects
            logging.warning(f"Cache write error in {self._func_name}: {e}")

        return result

    async def cache_batch_get(
        self,
        args_list: list[tuple[Any, ...]] | list[Any],
        kwargs_list: list[dict[str, Any]] | None = None,
    ) -> tuple[list[R | None], list[int]]:
        """Fetch multiple cache entries in a single Redis MGET.

        Args:
            args_list: Per-call positional-args tuples (or bare values).
            kwargs_list: Per-call keyword-arg dicts aligned to ``args_list``.
                Defaults to empty dicts when None.

        Returns:
            A ``(cached_values, missing_indices)`` tuple where ``cached_values``
            is aligned to ``args_list`` (None for each miss or decode error)
            and ``missing_indices`` lists the positions that need a fresh crawl.
        """
        if not args_list:
            return [], []

        normalized_args, kwargs_list = _normalize_batch_args(args_list, kwargs_list)

        config = get_cache_config()
        if not config.cache_enabled or config.storage_type != "redis":
            return [None] * len(normalized_args), list(range(len(normalized_args)))

        keys = [
            _generate_cache_key(self._prefix, self._schema_hash, *args, **kw)
            for args, kw in zip(normalized_args, kwargs_list, strict=False)
        ]

        try:
            redis_client = await get_result_cache_redis_client()
            cached_list = await redis_client.mget(keys)
        except RedisError as e:
            logging.warning(f"Cache batch read error in {self._func_name}: {e}")
            return [None] * len(normalized_args), list(range(len(normalized_args)))

        if not cached_list:
            return [None] * len(normalized_args), list(range(len(normalized_args)))

        results: list[R | None] = []
        missing: list[int] = []
        for idx, cached_data in enumerate(cached_list):
            if cached_data:
                try:
                    results.append(cast(R, json.loads(cached_data)))
                    continue
                except json.JSONDecodeError as e:
                    logging.warning(
                        f"Cache decode error in {self._func_name} for key {keys[idx]}: {e}"
                    )
            results.append(None)
            missing.append(idx)

        return results, missing

    async def cache_batch_set(
        self,
        args_list: list[tuple[Any, ...]] | list[Any],
        values: list[R | None],
        kwargs_list: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write multiple cache entries in a single Redis pipeline.

        None values in ``values`` are silently skipped. All non-None entries
        are queued as SETEX commands and flushed in one round-trip.
        Redis and serialization errors are logged and swallowed.

        Args:
            args_list: Per-call positional-args tuples (or bare values).
            values: Values aligned to ``args_list``. None entries are skipped.
            kwargs_list: Per-call keyword-arg dicts aligned to ``args_list``.
                Defaults to empty dicts when None.

        Raises:
            ValueError: If ``values`` or ``kwargs_list`` length differs from
                ``args_list``.
        """
        if not args_list:
            return

        normalized_args, kwargs_list = _normalize_batch_args(args_list, kwargs_list)

        if len(values) != len(normalized_args):
            raise ValueError("values length must match args_list length")

        config = get_cache_config()
        if not config.cache_enabled or config.storage_type != "redis":
            return

        cache_ttl = self._ttl if self._ttl is not None else 86400

        try:
            redis_client = await get_result_cache_redis_client()
            async with redis_client.pipeline(transaction=False) as pipe:
                for args, kw, value in zip(
                    normalized_args, kwargs_list, values, strict=False
                ):
                    if value is None:
                        continue
                    try:
                        cache_key = _generate_cache_key(
                            self._prefix, self._schema_hash, *args, **kw
                        )
                        serialized = json.dumps(value, ensure_ascii=False)
                        pipe.setex(cache_key, cache_ttl, serialized)
                    except TypeError as e:
                        logging.warning(f"Cache write error in {self._func_name}: {e}")
                await pipe.execute()
        except RedisError as e:
            logging.warning(f"Cache batch write error in {self._func_name}: {e}")


def cached_result(
    ttl: int | None = None,
    key_prefix: str | None = None,
    dependencies: list[Callable[..., Any]] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], _CachedFunction[P, R]]:
    """Decorator to cache async function results in Redis with automatic schema invalidation.

    The cache key includes a hash of the function's source code so that modifying
    the function (e.g. changing CSS selectors or extraction logic) automatically
    invalidates stale entries. The decorated function also gains ``cache_batch_get``
    and ``cache_batch_set`` helpers for bulk cache operations.

    Example:
        @cached_result(ttl=86400, key_prefix="animeplanet_anime")
        async def fetch_anime(slug: str) -> dict[str, Any] | None:
            ...  # expensive crawler operation

    Args:
        ttl: Time-to-live in seconds. Defaults to 86400 (24 h), independent of
            CacheConfig — result caching serves a different purpose than HTTP caching.
        key_prefix: Cache key prefix. Defaults to the decorated function's name.
        dependencies: Additional callables whose source is included in the schema
            hash. Any change to a listed callable invalidates the cache.

    Returns:
        A decorator that wraps the async function with Redis caching.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> _CachedFunction[P, R]:
        """Wrap an async function, returning a ``_CachedFunction`` with caching and batch helpers.

        Args:
            func: The async function to wrap.

        Returns:
            A ``_CachedFunction`` instance preserving the original function's
            signature, docstring, and ``__wrapped__`` attribute.
        """
        # Schema hash computed once at decoration time; changes when source changes.
        schema_hash = _compute_schema_hash(func, dependencies)
        prefix = key_prefix or getattr(func, "__name__", repr(func))
        return _CachedFunction(func, schema_hash, prefix, ttl)

    return decorator
