from datetime import timedelta
from unittest.mock import create_autospec

import grpc
import pytest
from common.config import EnrichmentConfig
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceNotConfiguredError,
    ServiceParseError,
    ServiceRateLimitedError,
)
from enrichment.sources.base.page_fetcher import FetchedPage, PageFetcher
from enrichment_proto.v1.enrichment_service_pb2 import FetchPageRequest
from enrichment_service.routes.fetch_page import build_page_fetchers, fetch_page
from enrichment_service.runtime import EnrichmentRuntime
from google.protobuf.duration_pb2 import Duration
from http_cache import HTTPCacheManager
from http_cache.network_request_pacer import NetworkRequestPacer
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _make_runtime() -> EnrichmentRuntime:
    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.network_request_pacer = NetworkRequestPacer()
    return EnrichmentRuntime(
        default_file_path="assets/seed_data/anime-offline-database.json",
        output_dir="temp",
        enrichment=EnrichmentConfig(),
        cache=cache,
    )


def _make_aborting_context() -> grpc.aio.ServicerContext:
    context = create_autospec(grpc.aio.ServicerContext, instance=True)
    context.abort.side_effect = grpc.RpcError()
    return context


def test_build_page_fetchers_registers_all_eight_providers() -> None:
    fetchers = build_page_fetchers(_make_runtime())
    assert set(fetchers.keys()) == {
        "anilist",
        "kitsu",
        "mal",
        "anime_planet",
        "anisearch",
        "anidb",
        "animeschedule",
        "syoboi",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_max_age", "expected_max_age"),
    [
        (None, None),
        (Duration(seconds=0), timedelta(0)),
        (
            Duration(seconds=30, nanos=500_000_000),
            timedelta(seconds=30, microseconds=500_000),
        ),
    ],
)
async def test_fetch_page_forwards_max_age_and_returns_stored_page(
    request_max_age: Duration | None,
    expected_max_age: timedelta | None,
) -> None:
    runtime = _make_runtime()
    page_key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    raw_response = PageResponse(
        request_name="main", content_type="application/json", body=b"{}"
    )
    fetcher = create_autospec(PageFetcher, instance=True)
    fetcher.fetch_page.return_value = FetchedPage(
        page_key=page_key, responses=(raw_response,), is_not_found=False
    )
    runtime.page_fetchers = {"anilist": fetcher}
    context = _make_aborting_context()
    request = (
        FetchPageRequest(page_key=page_key, max_age=request_max_age)
        if request_max_age is not None
        else FetchPageRequest(page_key=page_key)
    )

    response = await fetch_page(runtime, request, context)

    assert response.page.page_key == page_key
    assert list(response.page.responses) == [raw_response]
    fetcher.fetch_page.assert_awaited_once_with(page_key, max_age=expected_max_age)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_duration",
    [
        Duration(seconds=9223372036854775807),
        Duration(seconds=-9223372036854775808),
        Duration(seconds=1, nanos=1_000_000_000),
        Duration(seconds=1, nanos=-1),
        Duration(seconds=-1, nanos=1),
    ],
)
async def test_fetch_page_invalid_max_age_aborts_invalid_argument(
    invalid_duration: Duration,
) -> None:
    runtime = _make_runtime()
    page_key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    fetcher = create_autospec(PageFetcher, instance=True)
    runtime.page_fetchers = {"anilist": fetcher}
    context = _make_aborting_context()

    with pytest.raises(grpc.RpcError):
        await fetch_page(
            runtime,
            FetchPageRequest(page_key=page_key, max_age=invalid_duration),
            context,
        )

    assert context.abort.call_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT
    fetcher.fetch_page.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_factory", "fetcher_error", "expected_code"),
    [
        (lambda: FetchPageRequest(), None, grpc.StatusCode.INVALID_ARGUMENT),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="unknown",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="1",
                )
            ),
            None,
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="anilist",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="bad",
                )
            ),
            ValueError("numeric address"),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="anilist",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="21",
                )
            ),
            ServiceRateLimitedError(service="anilist", attempts=3),
            grpc.StatusCode.UNAVAILABLE,
        ),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="anidb",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="23",
                )
            ),
            ServiceBlockedError("banned", service="anidb"),
            grpc.StatusCode.UNAVAILABLE,
        ),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="animeschedule",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="cowboy-bebop",
                )
            ),
            ServiceNotConfiguredError(
                service="animeschedule",
                setting="ANIMESCHEDULE_TOKEN",
                reason="missing token",
            ),
            grpc.StatusCode.FAILED_PRECONDITION,
        ),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="syoboi",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="1",
                )
            ),
            ServiceParseError(service="syoboi", cause="bad xml"),
            grpc.StatusCode.UNAVAILABLE,
        ),
        (
            lambda: FetchPageRequest(
                page_key=PageKey(
                    provider="syoboi",
                    entity_type=EntityType.ENTITY_TYPE_ANIME,
                    address="1",
                )
            ),
            ServiceNetworkError(service="syoboi", cause="timeout"),
            grpc.StatusCode.UNAVAILABLE,
        ),
    ],
)
async def test_fetch_page_maps_errors_to_expected_grpc_codes(
    request_factory: object,
    fetcher_error: Exception | None,
    expected_code: grpc.StatusCode,
) -> None:
    runtime = _make_runtime()
    fetcher = create_autospec(PageFetcher, instance=True)
    fetcher.fetch_page.side_effect = fetcher_error
    runtime.page_fetchers = {
        "anilist": fetcher,
        "anidb": fetcher,
        "animeschedule": fetcher,
        "syoboi": fetcher,
    }
    context = _make_aborting_context()

    with pytest.raises(grpc.RpcError):
        await fetch_page(runtime, request_factory(), context)  # type: ignore[operator]

    assert context.abort.call_args.args[0] == expected_code
