"""
This module creates and exports a singleton instance of the HTTPCacheManager.

This ensures that a single, shared cache manager and its underlying connection pools
are used throughout the entire application, which is crucial for performance and
resource management in a concurrent environment.
"""

from .config import get_cache_config
from .manager import HTTPCacheManager

# Get cache configuration from environment variables
_cache_config = get_cache_config()

# Create the singleton instance of the HTTPCacheManager
http_cache_manager = HTTPCacheManager(_cache_config)
