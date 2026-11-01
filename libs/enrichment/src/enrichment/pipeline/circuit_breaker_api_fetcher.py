"""Pause a provider that keeps failing, and probe it until it answers again.

A bulk fetch, such as one over the whole seed, sends tens of thousands of
requests to each provider, so it must stop asking a site that blocks it. This
is the circuit breaker pattern: after failures the circuit opens and the queue
pauses, each probe tests it, and a probe that goes through closes it again.

Each item gives the rule, then the reason for it.

- After a run of failed requests in a row, the provider pauses and sends one
  request per probe interval until one goes through, then carries on. A
  provider that starts blocking is not sent the rest of its queue, and the run
  picks up again once the block lifts. Each probe uses the next page, so one
  broken page cannot stall the queue.
- After a set time paused, the fetcher gives up on the provider: every later
  request fails at once, without reaching the network. A site that is gone or
  an address that stays banned would otherwise keep a run probing for days;
  the pages left fail and are retried on the next run.
- Each provider has its own fetcher, so each pauses, resumes and gives up on
  its own clock.
- The gap between ordinary requests is not kept here but by the cache's
  network request pacer (``HTTPCacheManager.network_request_pacer``), which
  waits only for requests that reach the provider; an answer from the cache
  does not wait.

Seed cleaning and discovery fetch through it. ECHO-72 folds it into ``ApiFetcher``.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from common.config import EnrichmentConfig
from enrichment.pipeline.api_fetcher import ApiFetcher, HelperFactory
from http_cache import HTTPCacheManager

logger = logging.getLogger(__name__)


class CircuitBreakerApiFetcher(ApiFetcher):
    """Pause a provider queue that keeps failing.

    ``ProviderPageFetcher`` builds one ``ApiFetcher`` per provider queue, so
    each provider gets its own count of failures, its own pause and its own
    time paused.

    Each item reads ``<state>: <wait before a request>. <what ends the state>``.

    - Running: no wait of its own. A run of failures in a row pauses the
      provider.
    - Paused: each request waits the probe interval. A request that goes
      through resumes the provider; the time paused reaching the limit gives
      up on it.
    - Given up: no wait and no request; every request fails at once, for the
      next run to retry.
    """

    def __init__(
        self,
        config: EnrichmentConfig,
        cache: HTTPCacheManager,
        *,
        failures_before_pausing: int,
        seconds_between_probes: float,
        seconds_before_giving_up: float | None = None,
        helper_factories: dict[str, HelperFactory] | None = None,
    ) -> None:
        """Set up the circuit breaker fetcher.

        Args:
            config: Passed to ``ApiFetcher``, for the helpers that need it.
            cache: Passed to ``ApiFetcher``, for the helpers' HTTP sessions.
            failures_before_pausing: Failed requests in a row that pause the
                provider.
            seconds_between_probes: The wait before each request while the
                provider is paused.
            seconds_before_giving_up: The time paused, counted as the probe
                waits added up, after which the provider is given up on; None
                keeps probing until the queue runs out.
            helper_factories: Passed to ``ApiFetcher``; replaces its built-in
                helpers, such as with test helpers.
        """
        super().__init__(config, cache, helper_factories=helper_factories)
        self._failures_before_pausing = failures_before_pausing
        self._seconds_between_probes = seconds_between_probes
        self._seconds_before_giving_up = seconds_before_giving_up
        self._failures_in_a_row = 0
        self._is_paused = False
        self._seconds_paused = 0.0
        self._has_given_up = False

    async def fetch_all_data(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any]:
        """Fetch one page, after the probe interval while the provider is paused.

        Args:
            ids: The id field and page address, as ``ApiFetcher`` takes them.
            offline_data: The entry's title and sources.
            temp_dir: Passed to ``ApiFetcher``.
            skip_services: Passed to ``ApiFetcher``.
            only_services: The one service this queue fetches.
            fetch_characters: Passed to ``ApiFetcher``.
            fetch_episodes: Passed to ``ApiFetcher``.

        Returns:
            The service's record as ``ApiFetcher`` returns it.
        """
        service = (only_services or ["every service"])[0]
        if self._is_paused and self._is_pause_limit_reached():
            return self._fail_without_request(service)
        if self._is_paused:
            probe_at = datetime.now() + timedelta(seconds=self._seconds_between_probes)
            logger.warning(f"Waiting to probe {service} at {probe_at:%H:%M:%S}")
            await asyncio.sleep(self._seconds_between_probes)
            self._seconds_paused += self._seconds_between_probes
        records = await super().fetch_all_data(
            ids,
            offline_data,
            temp_dir,
            skip_services,
            only_services,
            fetch_characters=fetch_characters,
            fetch_episodes=fetch_episodes,
        )
        self._record_outcome(service, has_failed=bool(self.api_errors))
        return records

    def _record_outcome(self, service: str, *, has_failed: bool) -> None:
        """Count a request's failure or success, and pause or resume the provider.

        Args:
            service: The provider this queue fetches, for the log.
            has_failed: Whether the request failed.
        """
        if not has_failed:
            if self._is_paused:
                logger.warning(f"Resuming {service}: a probe went through")
            self._is_paused = False
            self._failures_in_a_row = 0
            self._seconds_paused = 0.0
            return
        self._failures_in_a_row += 1
        if not self._is_paused and (
            self._failures_in_a_row >= self._failures_before_pausing
        ):
            self._is_paused = True
            logger.warning(
                f"Pausing {service}: {self._failures_in_a_row} failures in a row; "
                f"probing every {self._seconds_between_probes:.0f}s"
            )

    def _is_pause_limit_reached(self) -> bool:
        """Report whether the provider has been paused as long as the limit allows."""
        return (
            self._seconds_before_giving_up is not None
            and self._seconds_paused >= self._seconds_before_giving_up
        )

    def _fail_without_request(self, service: str) -> dict[str, Any]:
        """Fail one request of a provider given up on, without reaching the network.

        Args:
            service: The provider this queue fetches.

        Returns:
            The service with no record, as ``ApiFetcher`` returns a failure; the
            reason is in ``api_errors``.
        """
        hours_paused = self._seconds_paused / 3600
        if not self._has_given_up:
            self._has_given_up = True
            logger.warning(
                f"Giving up on {service} after {hours_paused:.1f} hours paused; "
                "its remaining pages are retried on the next run"
            )
        self.api_timings = {}
        self.api_errors = {
            service: f"gave up after {hours_paused:.1f} hours paused; "
            "retried on the next run"
        }
        return {service: None}
