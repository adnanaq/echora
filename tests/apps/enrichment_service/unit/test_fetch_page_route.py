"""Unit tests for EnrichmentService FetchPage route and provider registry."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from common.config import EnrichmentConfig
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceRateLimitedError,
)
from enrichment.sources.base.page_fetcher import FetchedPage
from enrichment_proto.v1.enrichment_service_pb2 import FetchPageRequest
from enrichment_service.routes.fetch_page import build_page_fetchers, fetch_page
from enrichment_service.runtime import EnrichmentRuntime
from google.protobuf.duration_pb2 import Duration
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _make_runtime() -> EnrichmentRuntime:
    return EnrichmentRuntime(
        default_file_path="assets/seed_data/anime-offline-database.json",
        output_dir="temp",
        enrichment=EnrichmentConfig(),
        cache=MagicMock(),
    )


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
async def test_fetch_page_forwards_max_age_and_returns_stored_page() -> None:
    runtime = _make_runtime()
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    raw_resp = PageResponse(
        request_name="main", content_type="application/json", body=b"{}"
    )
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_page = AsyncMock(
        return_value=FetchedPage(
            page_key=key, responses=(raw_resp,), is_not_found=False
        )
    )
    runtime.page_fetchers = {"anilist": mock_fetcher}
    context = MagicMock()
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))

    response = await fetch_page(
        runtime,
        FetchPageRequest(page_key=key, max_age=Duration(seconds=30)),
        context,
    )

    assert response.page.page_key == key
    assert list(response.page.responses) == [raw_resp]
    mock_fetcher.fetch_page.assert_awaited_once_with(key, max_age=timedelta(seconds=30))


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
    ],
)
async def test_fetch_page_maps_errors_to_expected_grpc_codes(
    request_factory: object,
    fetcher_error: Exception | None,
    expected_code: grpc.StatusCode,
) -> None:
    runtime = _make_runtime()
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_page = AsyncMock(side_effect=fetcher_error)
    runtime.page_fetchers = {"anilist": mock_fetcher, "anidb": mock_fetcher}
    context = MagicMock()
    context.abort = AsyncMock(side_effect=grpc.RpcError())

    with pytest.raises(grpc.RpcError):
        await fetch_page(runtime, request_factory(), context)  # type: ignore[operator]

    assert context.abort.call_args.args[0] == expected_code
