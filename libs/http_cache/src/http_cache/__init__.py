"""HTTP caching infrastructure for enrichment pipeline."""

from .manager import HTTPCacheManager
from .result_cache import cached_result
from .setup import close_cache, configure_cache, get_cache

__all__ = [
    "HTTPCacheManager",
    "cached_result",
    "close_cache",
    "configure_cache",
    "get_cache",
]
