"""Fetch a site's pages over plain HTTP, spaced out and stopping at the first block.

For sites whose pages arrive complete without a browser. Two rules keep such a
crawler from getting its IP banned:

* requests to the site start at least ``min_interval`` seconds apart across the
  whole process, however many enrichments run at once;
* the first block response (HTTP 403, 423 or 429) or refused connection marks
  the site as blocked: that request raises ``ServiceBlockedError``, and so does
  every later one, without being sent. Nothing is retried. AniSearch, the site
  this was written for, answers a client it dislikes with 423 and then refuses
  every connection from that IP, so a retry only makes things worse.
"""

import asyncio
import logging
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass

import aiohttp
from enrichment.sources.base.exceptions import ServiceBlockedError

logger = logging.getLogger(__name__)

BLOCK_STATUSES = frozenset({403, 423, 429})


@dataclass(frozen=True)
class FetchedPage:
    """A page's HTML and the URL it ended up at after redirects."""

    url: str
    html: str


class PoliteHttpClient:
    """Plain-HTTP fetching for one site, with process-wide spacing and a block stop."""

    def __init__(
        self,
        site: str,
        headers: Mapping[str, str],
        *,
        min_interval: float,
        timeout: float = 30.0,
    ) -> None:
        self.site = site
        self.headers = dict(headers)
        self.min_interval = min_interval
        self.timeout = timeout
        self._blocked_reason: str | None = None
        self._next_request_at = 0.0
        # One lock per event loop: concurrent fetches queue on it, so each one
        # waits until _next_request_at before it is sent.
        self._request_spacing_locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Lock
        ] = weakref.WeakKeyDictionary()

    @property
    def blocked(self) -> bool:
        """Whether the site has blocked this process; no request is sent once it has."""
        return self._blocked_reason is not None

    async def fetch(self, url: str) -> FetchedPage | None:
        """Fetch ``url`` once ``min_interval`` has passed since the previous request.

        Args:
            url: Page to fetch.

        Returns:
            The page on HTTP 200, or ``None`` for any other answer that is not a
            block (such as 404), a dropped connection or a timeout, with a log line.

        Raises:
            ServiceBlockedError: If the site blocked this request or an earlier one.
        """
        if self._blocked_reason is not None:
            raise ServiceBlockedError(self._blocked_reason, service=self.site)
        await self._wait_for_request_slot()
        try:
            async with (
                aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as session,
                session.get(url) as response,
            ):
                if response.status in BLOCK_STATUSES:
                    self._block(f"HTTP {response.status} for {url}")
                if response.status != 200:
                    logger.warning(f"{self.site}: HTTP {response.status} for {url}")
                    return None
                return FetchedPage(url=str(response.url), html=await response.text())
        except aiohttp.ClientError as error:
            if isinstance(error, aiohttp.ClientConnectorError) and isinstance(
                error.os_error, ConnectionRefusedError
            ):
                self._block(f"connection refused for {url}")
            logger.warning(f"{self.site}: request failed for {url}: {error}")
            return None
        except TimeoutError:
            logger.warning(f"{self.site}: no answer within {self.timeout}s for {url}")
            return None

    async def _wait_for_request_slot(self) -> None:
        loop = asyncio.get_running_loop()
        spacing_lock = self._request_spacing_locks.get(loop)
        if spacing_lock is None:
            spacing_lock = self._request_spacing_locks[loop] = asyncio.Lock()
        async with spacing_lock:
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_request_at = time.monotonic() + self.min_interval

    def _block(self, reason: str) -> None:
        self._blocked_reason = reason
        logger.error(
            f"{self.site} blocked this client ({reason}); no more {self.site} "
            f"requests are sent until the process restarts"
        )
        raise ServiceBlockedError(reason, service=self.site)
