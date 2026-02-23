"""Utility functions for HTTP cache library."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def _mask_url_credentials(url: str) -> str:
    """
    Mask password in URL for safe logging.

    Replaces password component with '***' to prevent credential exposure in logs.

    Args:
        url: URL that may contain credentials (e.g., redis://user:password@host:port/db)

    Returns:
        URL with masked password (e.g., redis://user:***@host:port/db)
    """
    parsed = urlparse(url)
    if parsed.password:
        # Replace password with masked version
        masked_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return urlunparse(parsed._replace(netloc=masked_netloc))
    return url
