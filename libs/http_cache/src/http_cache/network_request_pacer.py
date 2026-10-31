"""Space out one process's network requests to each provider, so cache hits never wait.

A provider's rate limit counts the requests that reach it, not the answers the
cache gives, so the wait belongs where a request goes to the network.

Each item gives the rule, then the reason for it.

- A request waits only when it goes to the network. An HTTP session gets the
  wait through aiohttp's request-start hook (``build_trace_config``), which
  aiohttp runs only for a real request: inside ``CachedAiohttpSession`` hishel
  sends one only on a cache miss. ``@cached_result`` waits only before running
  its function on a miss.
- Each provider has its own gap, and different providers never wait for each
  other, so their queues still run side by side.
- A provider's gap is the larger of the default gap and its own minimum, such
  as AniDB's ``ANIDB_MIN_REQUEST_INTERVAL``, so a caller cannot lower a gap a
  provider needs.
- Each request reserves its time before waiting, so requests made at once to
  one provider still go out one gap apart.
- aiohttp starts a request's total timeout before its request-start hook, so
  the wait counts towards that timeout; provider timeouts are 30 to 60
  seconds, well above any gap.
"""

import asyncio
import time
from types import SimpleNamespace

import aiohttp


class NetworkRequestPacer:
    """The gap between one process's network requests to each provider."""

    def __init__(self) -> None:
        """Start with no gap for any provider."""
        self._default_seconds_between_requests = 0.0
        self._minimum_seconds_between_requests_by_service: dict[str, float] = {}
        self._next_request_at_by_service: dict[str, float] = {}

    def set_seconds_between_requests(self, seconds: float) -> None:
        """Set the gap every provider keeps between its network requests.

        Args:
            seconds: The gap; 0 leaves only providers' own minimums.
        """
        self._default_seconds_between_requests = seconds

    def set_minimum_seconds_between_requests(
        self, service: str, seconds: float
    ) -> None:
        """Set the smallest gap one provider keeps, whatever the default gap.

        Args:
            service: The provider, as the cache names it, such as ``"anidb"``.
            seconds: The provider's own minimum gap.
        """
        self._minimum_seconds_between_requests_by_service[service] = seconds

    def get_seconds_between_requests(self, service: str) -> float:
        """Return the gap one provider keeps between its network requests.

        Args:
            service: The provider, as the cache names it.

        Returns:
            The larger of the default gap and the provider's own minimum.
        """
        return max(
            self._default_seconds_between_requests,
            self._minimum_seconds_between_requests_by_service.get(service, 0.0),
        )

    async def wait_for_network_turn(self, service: str | None) -> None:
        """Wait until a network request to this provider may go out, and claim the slot.

        Args:
            service: The provider; None, for a function that names none, never
                waits.
        """
        if service is None:
            return
        seconds_between_requests = self.get_seconds_between_requests(service)
        if seconds_between_requests <= 0:
            return
        now = time.monotonic()
        request_at = max(now, self._next_request_at_by_service.get(service, now))
        self._next_request_at_by_service[service] = (
            request_at + seconds_between_requests
        )
        if request_at > now:
            await asyncio.sleep(request_at - now)

    def build_trace_config(self, service: str) -> aiohttp.TraceConfig:
        """Return an aiohttp hook making each network request to a provider wait its turn.

        Args:
            service: The provider the session sends requests to.

        Returns:
            A ``TraceConfig`` to pass in the session's ``trace_configs``.
        """
        trace_config = aiohttp.TraceConfig()

        async def wait_before_network_request(
            session: aiohttp.ClientSession,
            trace_context: SimpleNamespace,
            request_start: aiohttp.TraceRequestStartParams,
        ) -> None:
            await self.wait_for_network_turn(service)

        trace_config.on_request_start.append(wait_before_network_request)
        return trace_config
