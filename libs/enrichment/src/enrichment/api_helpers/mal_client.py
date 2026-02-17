"""Low-level MAL HTTP client.

This module provides a small client responsible for:
- Applying a shared, pre-request rate limiter.
- Performing HTTP GET requests.
- Retrying on HTTP 429 responses.
- Returning parsed JSON dictionaries (no business mapping).
"""

import asyncio
import logging
from typing import Any

from enrichment.api_helpers.mal_rate_limiter import (
    MalRateLimiter,
    get_shared_mal_rate_limiter,
)

logger = logging.getLogger(__name__)


class MalClient:
    """HTTP client for MAL API requests.

    Args:
        session: An aiohttp-style session (typically from `http_cache_manager`) that
            supports `session.get(...)` returning an async context manager.
        limiter: Optional limiter override. Defaults to the process-wide shared limiter.
        timeout_seconds: Total request timeout in seconds passed to `session.get`.
    """

    def __init__(
        self,
        *,
        session: Any,
        limiter: MalRateLimiter | Any | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._session = session
        self._limiter = limiter or get_shared_mal_rate_limiter()
        self._timeout = float(timeout_seconds)

    async def get_json(
        self, url: str, *, max_retries: int = 3
    ) -> dict[str, Any] | None:
        """GET a URL and decode a JSON object.

        This method acquires the shared limiter before each request attempt. If the
        response is HTTP 429, it sleeps and retries up to `max_retries`.

        Args:
            url: Absolute URL to fetch.
            max_retries: Maximum number of retries after a 429 response.

        Returns:
            The decoded JSON object if the response body is a dict; otherwise None.
        """
        retry = 0
        while True:
            await self._limiter.acquire()
            try:
                async with self._session.get(url, timeout=self._timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result if isinstance(result, dict) else None

                    if response.status == 429 and retry < max_retries:
                        retry += 1
                        await asyncio.sleep(5)
                        continue

                    logger.warning("MAL HTTP %s for %s", response.status, url)
                    return None
            except Exception:
                logger.exception("MAL request failed for %s", url)
                return None
