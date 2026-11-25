"""Unit tests for EnrichmentService.MergeEpisode route and episode FetchPage."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from enrichment.sources.mal.mal_fetcher import MalFetcher
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import MergeEpisodeRequest
from enrichment_service.routes.adapter import EnrichmentRoutes
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    PageKey,
    PageResponse,
    StoredPage,
)

if TYPE_CHECKING:
    from enrichment_service.runtime import EnrichmentRuntime


class _AbortCalledError(Exception):
    """Sentinel raised when mock context.abort is invoked."""


@pytest.mark.asyncio
async def test_merge_episode_route_validation_and_error_mapping() -> None:
    """MergeEpisode validates IDs and maps ValueError/Exception to gRPC status."""
    routes = EnrichmentRoutes(runtime=cast("EnrichmentRuntime", SimpleNamespace()))
    context = AsyncMock()
    context.abort.side_effect = _AbortCalledError()

    with pytest.raises(_AbortCalledError):
        await routes.MergeEpisode(
            MergeEpisodeRequest(episode_id="", anime_id="anime-1"),
            context,
        )
    context.abort.assert_awaited_with(
        grpc.StatusCode.INVALID_ARGUMENT, "episode_id is required"
    )

    context.abort.reset_mock()
    with pytest.raises(_AbortCalledError):
        await routes.MergeEpisode(
            MergeEpisodeRequest(episode_id="ep-1", anime_id=""),
            context,
        )
    context.abort.assert_awaited_with(
        grpc.StatusCode.INVALID_ARGUMENT, "anime_id is required"
    )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_episode.merge_episode_pages",
        side_effect=ValueError("invalid episode page"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeEpisode(
                MergeEpisodeRequest(episode_id="ep-1", anime_id="anime-1"),
                context,
            )
        context.abort.assert_awaited_with(
            grpc.StatusCode.INVALID_ARGUMENT, "invalid episode page"
        )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_episode.merge_episode_pages",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeEpisode(
                MergeEpisodeRequest(episode_id="ep-1", anime_id="anime-1"),
                context,
            )
        context.abort.assert_awaited_with(
            grpc.StatusCode.INTERNAL, "MergeEpisode failed"
        )


@pytest.mark.asyncio
async def test_merge_episode_end_to_end_over_grpc_transport() -> None:
    """End-to-end gRPC server -> stub call merges list entries and stored pages."""
    server = grpc.aio.server()
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        EnrichmentRoutes(runtime=cast("EnrichmentRuntime", SimpleNamespace())),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(channel)
            resp = await stub.MergeEpisode(
                MergeEpisodeRequest(
                    episode_id="ep-1",
                    anime_id="anime-1",
                    pages=[
                        StoredPage(
                            page_key=PageKey(
                                provider="kitsu",
                                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                                address="501",
                            ),
                            responses=[
                                PageResponse(
                                    request_name="main",
                                    content_type="application/json",
                                    body=json.dumps(
                                        {
                                            "data": {
                                                "id": "501",
                                                "type": "episodes",
                                                "attributes": {
                                                    "number": 1,
                                                    "canonicalTitle": "Kitsu Ep 1",
                                                },
                                            }
                                        }
                                    ).encode("utf-8"),
                                )
                            ],
                        )
                    ],
                    list_entries=[
                        EpisodeListEntry(
                            provider="mal", title="MAL Ep 1", is_filler=True
                        )
                    ],
                )
            )
            assert resp.episode.id == "ep-1"
            assert resp.episode.anime_id == "anime-1"
            assert resp.episode.episode_number == 1
            assert resp.episode.title == "MAL Ep 1"
            assert resp.episode.is_filler is True
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_mal_fetcher_supports_episode_page_key() -> None:
    """MalFetcher fetches raw HTML for ENTITY_TYPE_EPISODE PageKeys."""
    fetcher = MalFetcher()
    page_key = PageKey(
        provider="mal",
        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
        address="21:1",
    )
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(
        return_value=SimpleNamespace(browser=MagicMock())
    )
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=mock_session,
        ),
        patch(
            "enrichment.sources.mal.mal_fetcher._navigate_and_read",
            AsyncMock(
                return_value=(
                    "<html><h2>Ep 1</h2></html>",
                    "https://myanimelist.net/anime/21/One_Piece/episode/1",
                )
            ),
        ),
    ):
        fetched = await fetcher.fetch_page(page_key)
    assert fetched.is_not_found is False
    assert len(fetched.responses) == 1
    assert fetched.responses[0].request_name == "main"


@pytest.mark.asyncio
async def test_mal_fetcher_episode_error_and_not_found_branches() -> None:
    """MalFetcher validates episode address format and handles 404 HTML."""
    fetcher = MalFetcher()
    with pytest.raises(ValueError, match="must be '<anime_id>:<episode_number>'"):
        await fetcher.fetch_page(
            PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="bad-address",
            )
        )

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(
        return_value=SimpleNamespace(browser=MagicMock())
    )
    mock_session.__aexit__ = AsyncMock(return_value=None)
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=mock_session,
        ),
        patch(
            "enrichment.sources.mal.mal_fetcher._navigate_and_read",
            AsyncMock(
                return_value=(
                    '<div class="error404">404 Not Found</div>',
                    "https://myanimelist.net/anime/21/episode/999999",
                )
            ),
        ),
    ):
        not_found = await fetcher.fetch_page(
            PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="21:999999",
            )
        )
    assert not_found.is_not_found is True
    assert not_found.responses == ()
