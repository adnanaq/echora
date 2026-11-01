import logging
from typing import Any
from unittest.mock import call, create_autospec, patch

from common.config import EnrichmentConfig
from enrichment.pipeline import circuit_breaker_api_fetcher
from enrichment.pipeline.circuit_breaker_api_fetcher import CircuitBreakerApiFetcher
from enrichment.sources.base.base_helper import BaseEnrichmentHelper
from http_cache import HTTPCacheManager

PAYLOAD = {"anime": {"title": "One Piece"}}


class ScriptedHelper(BaseEnrichmentHelper):
    def __init__(self, answers: list[dict[str, Any] | Exception]) -> None:
        self.answers = answers
        self.request_count = 0

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any] | None:
        self.request_count += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _circuit_breaker_fetcher(
    helper: ScriptedHelper, seconds_before_giving_up: float | None = None
) -> CircuitBreakerApiFetcher:
    return CircuitBreakerApiFetcher(
        EnrichmentConfig(),
        create_autospec(HTTPCacheManager, instance=True),
        helper_factories={"kitsu": lambda config, cache: helper},
        failures_before_pausing=2,
        seconds_between_probes=1800,
        seconds_before_giving_up=seconds_before_giving_up,
    )


async def _fetch_times(fetcher: CircuitBreakerApiFetcher, request_count: int) -> None:
    for _ in range(request_count):
        await fetcher.fetch_all_data(
            {"kitsu_url": "https://kitsu.app/anime/12"}, {}, only_services=["kitsu"]
        )


async def test_fetch_all_data_running_provider_never_waits() -> None:
    fetcher = _circuit_breaker_fetcher(ScriptedHelper([PAYLOAD, PAYLOAD, PAYLOAD]))
    with patch.object(
        circuit_breaker_api_fetcher.asyncio, "sleep", autospec=True
    ) as sleep:
        await _fetch_times(fetcher, 3)
    assert sleep.call_args_list == []


async def test_fetch_all_data_failures_in_row_pause_until_probe_goes_through() -> None:
    blocked = RuntimeError("HTTP 429")
    fetcher = _circuit_breaker_fetcher(
        ScriptedHelper([blocked, blocked, blocked, PAYLOAD, PAYLOAD])
    )
    with patch.object(
        circuit_breaker_api_fetcher.asyncio, "sleep", autospec=True
    ) as sleep:
        await _fetch_times(fetcher, 5)
    assert sleep.call_args_list == [call(1800), call(1800)]


async def test_fetch_all_data_paused_past_limit_gives_up_without_request(
    caplog,
) -> None:
    blocked = RuntimeError("HTTP 429")
    helper = ScriptedHelper([blocked, blocked, blocked, blocked])
    fetcher = _circuit_breaker_fetcher(helper, seconds_before_giving_up=3600)
    with (
        patch.object(
            circuit_breaker_api_fetcher.asyncio, "sleep", autospec=True
        ) as sleep,
        caplog.at_level(logging.WARNING, logger=circuit_breaker_api_fetcher.__name__),
    ):
        await _fetch_times(fetcher, 6)
    assert (
        helper.request_count,
        sleep.call_args_list,
        list(fetcher.api_errors),
        sum("Giving up on kitsu" in message for message in caplog.messages),
    ) == (4, [call(1800), call(1800)], ["kitsu"], 1)


async def test_fetch_all_data_probe_going_through_restarts_pause_time() -> None:
    blocked = RuntimeError("HTTP 429")
    helper = ScriptedHelper(
        [blocked, blocked, blocked, PAYLOAD, blocked, blocked, blocked, PAYLOAD]
    )
    fetcher = _circuit_breaker_fetcher(helper, seconds_before_giving_up=3600)
    with patch.object(
        circuit_breaker_api_fetcher.asyncio, "sleep", autospec=True
    ) as sleep:
        await _fetch_times(fetcher, 8)
    assert (helper.request_count, sleep.call_args_list) == (
        8,
        [call(1800), call(1800), call(1800), call(1800)],
    )
