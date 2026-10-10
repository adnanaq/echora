import logging
from typing import Any
from unittest.mock import patch

import pytest
from common.config import EnrichmentConfig
from enrichment.pipeline.api_fetcher import ApiFetcher
from enrichment.sources.anidb.anidb_helper import AniDBHelper
from enrichment.sources.anilist.anilist_helper import AniListHelper
from enrichment.sources.anime_planet.anime_planet_helper import AnimePlanetHelper
from enrichment.sources.animeschedule.animeschedule_helper import AnimescheduleHelper
from enrichment.sources.anisearch.anisearch_helper import AniSearchHelper
from enrichment.sources.base.base_helper import BaseEnrichmentHelper
from enrichment.sources.kitsu.kitsu_helper import KitsuHelper
from enrichment.sources.mal.mal_helper import MalHelper
from http_cache import HTTPCacheManager, get_cache

PAYLOAD = {
    "anime": {"title": "One Piece"},
    "episodes": [],
    "characters": [],
    "extras": {},
}


class RecordingHelper(BaseEnrichmentHelper):
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.requests: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.closed = False

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any] | None:
        self.requests.append(
            (
                (ids, offline_data, temp_dir),
                {
                    "fetch_characters": fetch_characters,
                    "fetch_episodes": fetch_episodes,
                },
            )
        )
        if self.error is not None:
            raise self.error
        return self.result

    async def close(self) -> None:
        self.closed = True


def _fetcher(**helpers: RecordingHelper) -> ApiFetcher:
    return ApiFetcher(
        EnrichmentConfig(),
        get_cache(),
        helper_factories={
            name: lambda config, cache, helper=helper: helper
            for name, helper in helpers.items()
        },
    )


async def test_api_fetcher_context_manager_returns_itself() -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())

    async with fetcher as entered:
        assert entered is fetcher


async def test_api_fetcher_context_manager_propagates_exception() -> None:
    with pytest.raises(ValueError, match="Test error"):
        async with ApiFetcher(EnrichmentConfig(), get_cache()):
            raise ValueError("Test error")


@pytest.mark.parametrize(
    ("service", "helper_class"),
    [
        ("anilist", AniListHelper),
        ("kitsu", KitsuHelper),
        ("anidb", AniDBHelper),
        ("anime_planet", AnimePlanetHelper),
        ("anisearch", AniSearchHelper),
        ("animeschedule", AnimescheduleHelper),
        ("mal", MalHelper),
    ],
)
def test_build_helpers_registry_builds_each_service_helper(
    service: str, helper_class: type
) -> None:
    helpers = ApiFetcher(EnrichmentConfig(), get_cache())._build_helpers()

    assert isinstance(helpers[service], helper_class)


@pytest.mark.parametrize("service", ["anilist", "kitsu", "animeschedule"])
def test_build_helpers_http_helpers_get_fetcher_cache(service: str) -> None:
    cache = get_cache()

    helper = ApiFetcher(EnrichmentConfig(), cache)._build_helpers()[service]

    assert helper._cache is cache


def test_build_helpers_anidb_gets_fetcher_config_and_cache() -> None:
    config = EnrichmentConfig(anidb_client="registered", anidb_max_retries=1)
    cache = get_cache()

    helper = ApiFetcher(config, cache)._build_helpers()["anidb"]

    assert helper._cache is cache
    assert helper.client_name == "registered"
    assert helper.max_retries == 1


def test_build_helpers_only_filter_returns_listed_services() -> None:
    helpers = ApiFetcher(EnrichmentConfig(), get_cache())._build_helpers(
        only_services=["kitsu", "anidb"]
    )

    assert set(helpers) == {"kitsu", "anidb"}


def test_build_helpers_skip_filter_leaves_out_skipped_services() -> None:
    helpers = ApiFetcher(EnrichmentConfig(), get_cache())._build_helpers(
        skip_services=["anidb", "anisearch"]
    )

    assert set(helpers) == set(ApiFetcher._REGISTRY) - {"anidb", "anisearch"}


def test_build_helpers_each_call_returns_new_instances() -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())

    first = fetcher._build_helpers(only_services=["kitsu"])
    second = fetcher._build_helpers(only_services=["kitsu"])

    assert first["kitsu"] is not second["kitsu"]


def test_build_helpers_given_factories_replace_registry() -> None:
    built: list[tuple[EnrichmentConfig, HTTPCacheManager]] = []
    config = EnrichmentConfig()
    cache = get_cache()

    def build(given_config: EnrichmentConfig, given_cache: HTTPCacheManager):
        built.append((given_config, given_cache))
        return RecordingHelper()

    helpers = ApiFetcher(
        config, cache, helper_factories={"custom_svc": build}
    )._build_helpers()

    assert set(helpers) == {"custom_svc"}
    assert built == [(config, cache)]


@pytest.mark.parametrize(
    ("service", "only", "skip", "expected"),
    [
        ("mal", ["mal", "kitsu"], None, True),
        ("anidb", ["mal", "kitsu"], None, False),
        ("mal", None, ["mal"], False),
        ("kitsu", None, ["mal"], True),
        ("anything", None, None, True),
        ("mal", ["mal"], ["mal"], True),
    ],
)
def test_should_include_filters_return_expected(
    service: str, only: list[str] | None, skip: list[str] | None, expected: bool
) -> None:
    assert ApiFetcher._should_include(service, only, skip) is expected


async def test_fetch_all_data_returns_normalized_payload_per_service() -> None:
    fetcher = _fetcher(
        mal=RecordingHelper(PAYLOAD),
        anisearch=RecordingHelper(
            {"title": "One Piece", "episodes": [], "characters": []}
        ),
    )

    result = await fetcher.fetch_all_data({}, {"title": "One Piece"})

    assert result == {"mal": PAYLOAD, "anisearch": PAYLOAD}


async def test_fetch_all_data_passes_inputs_and_flags_to_each_helper() -> None:
    helper = RecordingHelper(PAYLOAD)
    ids = {"anidb_url": "https://anidb.net/anime/1"}
    offline = {"title": "Test", "sources": []}

    await _fetcher(anidb=helper).fetch_all_data(
        ids, offline, temp_dir="temp/Test_agent1", fetch_characters=False
    )

    assert helper.requests == [
        (
            (ids, offline, "temp/Test_agent1"),
            {"fetch_characters": False, "fetch_episodes": True},
        )
    ]


async def test_fetch_all_data_fetch_episodes_false_passed_to_helpers() -> None:
    helper = RecordingHelper(PAYLOAD)

    await _fetcher(mal=helper).fetch_all_data({}, {}, fetch_episodes=False)

    assert helper.requests[0][1] == {"fetch_characters": True, "fetch_episodes": False}


async def test_fetch_all_data_only_services_runs_listed_services() -> None:
    kitsu = RecordingHelper(PAYLOAD)
    anidb = RecordingHelper(PAYLOAD)

    result = await _fetcher(kitsu=kitsu, anidb=anidb).fetch_all_data(
        {}, {}, only_services=["kitsu"]
    )

    assert set(result) == {"kitsu"}
    assert anidb.requests == []


async def test_fetch_all_data_skip_services_leaves_out_skipped() -> None:
    kitsu = RecordingHelper(PAYLOAD)
    anidb = RecordingHelper(PAYLOAD)

    result = await _fetcher(kitsu=kitsu, anidb=anidb).fetch_all_data(
        {}, {}, skip_services=["kitsu"]
    )

    assert set(result) == {"anidb"}
    assert kitsu.requests == []


async def test_fetch_all_data_failing_helper_records_error_and_keeps_others() -> None:
    fetcher = _fetcher(
        mal=RecordingHelper(PAYLOAD),
        kitsu=RecordingHelper(error=RuntimeError("Network error")),
    )

    result = await fetcher.fetch_all_data({}, {})

    assert result == {"mal": PAYLOAD, "kitsu": None}
    assert "Network error" in fetcher.api_errors["kitsu"]
    assert "mal" in fetcher.api_timings


async def test_fetch_all_data_none_result_returns_none_without_error() -> None:
    fetcher = _fetcher(mal=RecordingHelper(None))

    result = await fetcher.fetch_all_data({}, {})

    assert result == {"mal": None}
    assert "mal" in fetcher.api_timings
    assert fetcher.api_errors == {}


async def test_fetch_all_data_closes_helpers_after_run() -> None:
    helper = RecordingHelper(PAYLOAD)

    await _fetcher(mal=helper).fetch_all_data({}, {})

    assert helper.closed is True


async def test_fetch_all_data_gather_error_raises_after_closing_helpers() -> None:
    helper = RecordingHelper(PAYLOAD)
    fetcher = _fetcher(mal=helper)

    with patch.object(
        fetcher, "_gather", autospec=True, side_effect=RuntimeError("gather boom")
    ):
        with pytest.raises(RuntimeError, match="gather boom"):
            await fetcher.fetch_all_data({}, {})

    assert helper.closed is True


async def test_fetch_all_data_resets_metrics_from_previous_run() -> None:
    fetcher = _fetcher(mal=RecordingHelper(PAYLOAD))
    fetcher.api_timings = {"stale": 9.9}
    fetcher.api_errors = {"broken": "old error"}

    await fetcher.fetch_all_data({}, {})

    assert set(fetcher.api_timings) == {"mal"}
    assert fetcher.api_errors == {}


async def test_fetch_all_data_each_run_builds_new_helpers() -> None:
    built: list[RecordingHelper] = []

    def build(config: EnrichmentConfig, cache: HTTPCacheManager) -> RecordingHelper:
        built.append(RecordingHelper(PAYLOAD))
        return built[-1]

    fetcher = ApiFetcher(
        EnrichmentConfig(), get_cache(), helper_factories={"mal": build}
    )

    await fetcher.fetch_all_data({}, {})
    await fetcher.fetch_all_data({}, {})

    assert len(built) == 2
    assert built[0] is not built[1]


async def test_gather_exception_recorded_and_result_none() -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())

    async def succeed() -> dict[str, bool]:
        return {"ok": True}

    async def fail() -> None:
        raise RuntimeError("api down")

    result = await fetcher._gather([("good", succeed()), ("bad", fail())])

    assert result == {"good": {"ok": True}, "bad": None}
    assert "api down" in fetcher.api_errors["bad"]


async def test_gather_none_result_kept_without_error() -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())

    async def nothing() -> None:
        return None

    assert await fetcher._gather([("svc", nothing())]) == {"svc": None}
    assert fetcher.api_errors == {}


def test_log_performance_metrics_logs_timings_errors_and_success_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())
    fetcher.api_timings = {"mal": 1.5, "kitsu": 2.0}
    fetcher.api_errors = {"anidb": "timeout"}

    with caplog.at_level(logging.INFO, logger="enrichment.pipeline.api_fetcher"):
        fetcher._log_performance_metrics(3.5)

    assert "Total Time: 3.50s" in caplog.text
    assert "mal: 1.50s" in caplog.text
    assert "anidb: timeout" in caplog.text
    assert "Success Rate: 66.7%" in caplog.text


def test_log_performance_metrics_no_services_logs_zero_success_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())

    with caplog.at_level(logging.INFO, logger="enrichment.pipeline.api_fetcher"):
        fetcher._log_performance_metrics(0.0)

    assert "Success Rate: 0.0%" in caplog.text


def test_log_performance_metrics_no_errors_logs_full_success_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = ApiFetcher(EnrichmentConfig(), get_cache())
    fetcher.api_timings = {"mal": 1.0, "kitsu": 0.5}

    with caplog.at_level(logging.INFO, logger="enrichment.pipeline.api_fetcher"):
        fetcher._log_performance_metrics(1.5)

    assert "Success Rate: 100.0%" in caplog.text
    assert "API Errors" not in caplog.text
