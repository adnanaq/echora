"""Shared browser and crawler configuration for all crawlers.

Two factory variants:
- In-process (get_browser_config / get_crawler_config): return Pydantic objects
  for crawlers using AsyncWebCrawler directly (AnimePlanet, AniDB, AniSearch).
- Docker dict (get_docker_browser_config / get_docker_crawler_config): return
  serialized dicts for crawlers using the crawl4ai Docker REST API (MAL).

Rate limiting:
- CrawlerRateLimiter — generic async rate limiter for any crawler.
  Import from here; MAL-specific wrappers in mal_rate_limiter.py re-export it.

All crawlers import constants and factories from here — no duplication.
"""

import asyncio
import time
from collections import deque
from functools import lru_cache
from typing import Any

from crawl4ai import BrowserConfig, CrawlerRunConfig


class CrawlerRateLimiter:
    """Async rate limiter for crawler batch submissions.

    Throttles request *start times* to respect:
    - A minimum interval between submissions (min_interval_seconds).
    - A maximum number of submissions per rolling 60-second window (max_per_minute).

    Safe to share across coroutines in a single process.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = 0.5,
        max_per_minute: int = 60,
    ) -> None:
        self._min_interval = float(min_interval_seconds)
        self._max_per_minute = int(max_per_minute)
        self._lock = asyncio.Lock()
        self._last: float | None = None
        self._recent: deque[float] = deque()

    async def acquire(self) -> None:
        """Block until a new submission is allowed under the configured limits."""
        async with self._lock:
            now = time.time()

            cutoff = now - 60.0
            while self._recent and self._recent[0] <= cutoff:
                self._recent.popleft()

            if self._max_per_minute > 0 and len(self._recent) >= self._max_per_minute:
                oldest = self._recent[0]
                sleep_for = max(0.0, (oldest + 60.0) - now)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                    now = time.time()
                    cutoff = now - 60.0
                    while self._recent and self._recent[0] <= cutoff:
                        self._recent.popleft()

            if self._last is not None and self._min_interval > 0:
                sleep_for = (self._last + self._min_interval) - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                    now = time.time()

            self._last = now
            self._recent.append(now)


@lru_cache(maxsize=1)
def get_ap_rate_limiter() -> CrawlerRateLimiter:
    """Shared rate limiter for Anime-Planet character detail crawls.

    5 s minimum between batch submissions prevents Cloudflare 403s on
    large casts (e.g. One Piece: 1088 characters, 54 batches).
    """
    return CrawlerRateLimiter(min_interval_seconds=5.0, max_per_minute=12)


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

DEFAULT_VIEWPORT: tuple[int, int] = (1920, 1080)
DEFAULT_PAGE_TIMEOUT: int = 90_000


# =============================================================================
# In-process factories (AsyncWebCrawler)
# =============================================================================


def get_browser_config(
    *,
    headless: bool = True,
    stealth: bool = True,
    verbose: bool = False,
    extra_headers: dict[str, str] | None = None,
) -> BrowserConfig:
    """BrowserConfig for in-process AsyncWebCrawler."""
    headers = {**DEFAULT_HEADERS, **(extra_headers or {})}
    return BrowserConfig(
        headless=headless,
        verbose=verbose,
        enable_stealth=stealth,
        headers=headers,
        viewport_width=DEFAULT_VIEWPORT[0],
        viewport_height=DEFAULT_VIEWPORT[1],
    )


def get_crawler_config(
    extraction_strategy: Any,
    *,
    wait_until: str = "load",
    simulate_user: bool = True,
    override_navigator: bool = True,
    magic: bool = True,
    delay: float | None = None,
    page_timeout: int = DEFAULT_PAGE_TIMEOUT,
) -> CrawlerRunConfig:
    """CrawlerRunConfig for in-process AsyncWebCrawler."""
    kwargs: dict[str, Any] = {
        "extraction_strategy": extraction_strategy,
        "simulate_user": simulate_user,
        "override_navigator": override_navigator,
        "magic": magic,
        "wait_until": wait_until,
        "page_timeout": page_timeout,
    }
    if delay is not None:
        kwargs["delay_before_return_html"] = delay
    return CrawlerRunConfig(**kwargs)


# =============================================================================
# Docker dict factories (crawl4ai Docker REST API)
# =============================================================================


def get_docker_browser_config(
    extra_headers: dict[str, str] | None = None,
    *,
    stealth: bool = True,
) -> dict[str, Any]:
    """Serialized BrowserConfig dict for the crawl4ai Docker REST API."""
    headers = {**DEFAULT_HEADERS, **(extra_headers or {})}
    return {
        "type": "BrowserConfig",
        "params": {
            "headless": True,
            "verbose": False,
            "enable_stealth": stealth,
            "headers": headers,
            "viewport_width": DEFAULT_VIEWPORT[0],
            "viewport_height": DEFAULT_VIEWPORT[1],
        },
    }


def get_docker_crawler_config(
    schema: dict[str, Any],
    *,
    wait_until: str = "domcontentloaded",
    delay: float | None = None,
    page_timeout: int = DEFAULT_PAGE_TIMEOUT,
    extraction_type: str = "JsonXPathExtractionStrategy",
) -> dict[str, Any]:
    """Serialized CrawlerRunConfig dict for the crawl4ai Docker REST API."""
    params: dict[str, Any] = {
        "extraction_strategy": {
            "type": extraction_type,
            "params": {"schema": schema},
        },
        "simulate_user": True,
        "override_navigator": True,
        "magic": True,
        "wait_until": wait_until,
        "page_timeout": page_timeout,
    }
    if delay is not None:
        params["delay_before_return_html"] = delay
    return {"type": "CrawlerRunConfig", "params": params}
