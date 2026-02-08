"""MAL request rate limiting utilities.

This module provides a process-wide shared async rate limiter used by all MAL API calls.
It is designed to be acquired *before* requests, so all endpoints and helper instances
share a single quota budget.
"""

import asyncio
import time
from collections import deque
from functools import lru_cache


class MalRateLimiter:
    """Asynchronous rate limiter for MAL requests.

    This limiter throttles request *start times* to respect:
    - A minimum interval between requests (e.g., 0.5s).
    - A maximum number of requests per rolling 60-second window.

    The limiter is safe to share across tasks in a single process.

    Args:
        min_interval_seconds: Minimum spacing between request starts.
        max_per_minute: Maximum request starts allowed in the last 60 seconds.
    """

    def __init__(
        self, *, min_interval_seconds: float = 0.5, max_per_minute: int = 60
    ) -> None:
        self._min_interval = float(min_interval_seconds)
        self._max_per_minute = int(max_per_minute)
        self._lock = asyncio.Lock()
        self._last: float | None = None
        self._recent: deque[float] = deque()

    async def acquire(self) -> None:
        """Wait until a new request can be started under the configured limits.

        This method blocks cooperatively (via `asyncio.sleep`) and is safe to call
        concurrently from multiple tasks.

        Returns:
            None
        """
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


DEFAULT_MIN_INTERVAL_SECONDS = 0.5
DEFAULT_MAX_PER_MINUTE = 60


@lru_cache(maxsize=1)
def get_shared_mal_rate_limiter() -> "MalRateLimiter":
    """Return a process-wide shared limiter instance for all MAL requests.

    This is the default limiter used by `MalClient` and therefore all higher-level
    helpers unless explicitly overridden.

    Returns:
        MalRateLimiter: A singleton limiter for the current Python process.
    """
    return MalRateLimiter(
        min_interval_seconds=DEFAULT_MIN_INTERVAL_SECONDS,
        max_per_minute=DEFAULT_MAX_PER_MINUTE,
    )
