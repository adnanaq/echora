from typing import Any
from unittest.mock import call, create_autospec, patch

from common.config import EnrichmentConfig
from enrichment.pipeline import pausing_api_fetcher
from enrichment.pipeline.pausing_api_fetcher import PausingApiFetcher
from enrichment.sources.base.base_helper import BaseEnrichmentHelper
from http_cache import HTTPCacheManager

PAYLOAD = {"anime": {"title": "One Piece"}}


class ScriptedHelper(BaseEnrichmentHelper):
    def __init__(self, answers: list[dict[str, Any] | Exception]) -> None:
        self.answers = answers

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any] | None:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _pausing_fetcher(answers: list[dict[str, Any] | Exception]) -> PausingApiFetcher:
    helper = ScriptedHelper(answers)
    return PausingApiFetcher(
        EnrichmentConfig(),
        create_autospec(HTTPCacheManager, instance=True),
        helper_factories={"kitsu": lambda config, cache: helper},
        failures_before_pausing=2,
        seconds_between_probes=1800,
    )


async def _fetch_times(fetcher: PausingApiFetcher, request_count: int) -> None:
    for _ in range(request_count):
        await fetcher.fetch_all_data(
            {"kitsu_url": "https://kitsu.app/anime/12"}, {}, only_services=["kitsu"]
        )


async def test_fetch_all_data_running_provider_never_waits() -> None:
    fetcher = _pausing_fetcher([PAYLOAD, PAYLOAD, PAYLOAD])
    with patch.object(pausing_api_fetcher.asyncio, "sleep", autospec=True) as sleep:
        await _fetch_times(fetcher, 3)
    assert sleep.call_args_list == []


async def test_fetch_all_data_failures_in_row_pause_until_probe_goes_through() -> None:
    blocked = RuntimeError("HTTP 429")
    fetcher = _pausing_fetcher([blocked, blocked, blocked, PAYLOAD, PAYLOAD])
    with patch.object(pausing_api_fetcher.asyncio, "sleep", autospec=True) as sleep:
        await _fetch_times(fetcher, 5)
    assert sleep.call_args_list == [call(1800), call(1800)]
