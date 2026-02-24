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
        # Split on the first '@' to isolate host:port from credentials.
        # Avoids decoded vs. encoded mismatch that makes string-replace unreliable
        # (parsed.password is URL-decoded; parsed.netloc retains percent-encoding).
        host_part = parsed.netloc.split("@", 1)[-1]
        masked_netloc = f"{parsed.username}:***@{host_part}"
        return urlunparse(parsed._replace(netloc=masked_netloc))
    return url
