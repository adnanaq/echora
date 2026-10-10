"""
HTTP Cache - Redis-backed HTTP caching library.

Generic caching infrastructure for HTTP requests and function results.
Provides RFC 9111 compliant HTTP caching via Hishel and custom result caching.

Takes its configuration from common; the app sets it up with configure_cache.
"""

python_distribution(
    name="http_cache_dist",
    dependencies=[
        "//libs/http_cache/src/http_cache",
    ],
    provides=python_artifact(
        name="echora-http-cache",
        version="0.1.0",
        description="Redis-backed HTTP caching with aiohttp, Hishel, and result caching decorator",
    ),
    generate_setup=True,
)
