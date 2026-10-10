"""The process-wide cache, configured once by the app at startup.

Apps call ``configure_cache`` after ``get_settings()`` and ``close_cache`` on
shutdown. Code that runs inside the app, such as crawler functions, looks the
cache up with ``get_cache`` when it runs, never at import.
"""

import logging

from common.config import CacheConfig, RedisConfig

from .manager import HTTPCacheManager

logger = logging.getLogger(__name__)

_manager: HTTPCacheManager | None = None


def configure_cache(cache: CacheConfig, redis: RedisConfig) -> HTTPCacheManager:
    """Set the cache every caller of ``get_cache`` gets, replacing any earlier one.

    Args:
        cache: What to cache and for how long, usually ``settings.cache``.
        redis: Where Redis is, usually ``settings.redis``.

    Returns:
        The configured manager, for objects that take it in their constructor.

    Examples:
        >>> settings = get_settings()
        >>> cache = configure_cache(settings.cache, settings.redis)
    """
    global _manager
    _manager = HTTPCacheManager(cache, redis)
    return _manager


def get_cache() -> HTTPCacheManager:
    """Return the configured cache, or a disabled one if nothing configured it.

    Returns:
        The manager set by ``configure_cache``. Without one, such as in tests
        or a library used on its own, a manager with caching off.
    """
    global _manager
    if _manager is None:
        logger.info("Cache not configured; running uncached")
        # model_construct skips reading environment variables: an unconfigured
        # cache must not pick up settings the app never chose.
        _manager = HTTPCacheManager(
            CacheConfig.model_construct(cache_enabled=False),
            RedisConfig.model_construct(),
        )
    return _manager


async def close_cache() -> None:
    """Close the cache's Redis clients; call once when the app shuts down."""
    if _manager is not None:
        await _manager.close()
