from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Self
from unittest.mock import create_autospec, patch

import grpc
import pytest
from enrichment.sources.mal import mal_fetcher
from enrichment.sources.mal.mal_fetcher import MalFetcher
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import (
    MergeEpisodeRequest,
    MergeEpisodeResponse,
)
from enrichment_service.routes.adapter import EnrichmentRoutes
from enrichment_service.runtime import EnrichmentRuntime
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    PageKey,
    PageResponse,
    StoredPage,
)


class _AbortCalledError(Exception):
    pass


class _StubBrowserSession:
    def __init__(self) -> None:
        self.browser = object()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


def _build_routes() -> EnrichmentRoutes:
    return EnrichmentRoutes(runtime=create_autospec(EnrichmentRuntime, instance=True))


def _build_context() -> grpc.aio.ServicerContext:
    context = create_autospec(grpc.aio.ServicerContext, instance=True)
    context.abort.side_effect = _AbortCalledError()
    return context


@pytest.mark.asyncio
async def test_merge_episode_invalid_ids_and_errors_map_to_expected_status() -> None:
    routes = _build_routes()
    context = _build_context()

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

    for error in (ValueError("invalid episode page"), TypeError("bad kitsu shape")):
        context.abort.reset_mock()
        with patch(
            "enrichment_service.routes.merge_episode.merge_episode_pages",
            autospec=True,
            side_effect=error,
        ):
            with pytest.raises(_AbortCalledError):
                await routes.MergeEpisode(
                    MergeEpisodeRequest(episode_id="ep-1", anime_id="anime-1"),
                    context,
                )
            context.abort.assert_awaited_with(
                grpc.StatusCode.INVALID_ARGUMENT, str(error)
            )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_episode.merge_episode_pages",
        autospec=True,
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
async def test_merge_episode_malformed_kitsu_alongside_good_mal_aborts_invalid_argument() -> (
    None
):
    routes = _build_routes()
    context = _build_context()
    mal_html = Path(
        "tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_1.html"
    ).read_bytes()
    request = MergeEpisodeRequest(
        episode_id="ep-1",
        anime_id="anime-21",
        episode_number=1,
        pages=[
            StoredPage(
                page_key=PageKey(
                    provider="mal",
                    entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                    address="21:1",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="text/html",
                        body=mal_html,
                    )
                ],
            ),
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
                        body=b'{"data": null}',
                    )
                ],
            ),
        ],
    )
    with pytest.raises(_AbortCalledError):
        await routes.MergeEpisode(request, context)
    assert context.abort.await_args is not None
    assert context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_merge_episode_executes_off_event_loop_thread() -> None:
    routes = _build_routes()
    context = _build_context()
    loop_thread_id = threading.get_ident()
    observed_thread_ids: list[int] = []

    def _record_thread(request: MergeEpisodeRequest) -> MergeEpisodeResponse:
        del request
        observed_thread_ids.append(threading.get_ident())
        return MergeEpisodeResponse()

    with patch(
        "enrichment_service.routes.merge_episode.merge_episode_pages",
        autospec=True,
        side_effect=_record_thread,
    ):
        await routes.MergeEpisode(
            MergeEpisodeRequest(episode_id="ep-1", anime_id="anime-1"),
            context,
        )

    assert observed_thread_ids
    assert observed_thread_ids[0] != loop_thread_id


@pytest.mark.asyncio
async def test_merge_episode_grpc_transport_merges_pages_and_list_entries() -> None:
    server = grpc.aio.server()
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        _build_routes(),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(channel)
            response = await stub.MergeEpisode(
                MergeEpisodeRequest(
                    episode_id="ep-1",
                    anime_id="anime-1",
                    episode_number=1,
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
            assert response.episode.id == "ep-1"
            assert response.episode.anime_id == "anime-1"
            assert response.episode.episode_number == 1
            assert response.episode.title == "MAL Ep 1"
            assert response.episode.is_filler is True
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_mal_fetcher_episode_page_key_returns_html_response() -> None:
    fetcher = MalFetcher()
    page_key = PageKey(
        provider="mal",
        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
        address="21:1",
    )
    with (
        patch.object(
            mal_fetcher,
            "browser_session",
            autospec=True,
            return_value=_StubBrowserSession(),
        ),
        patch.object(
            mal_fetcher,
            "_navigate_and_read",
            autospec=True,
            return_value=(
                "<html><h2>Ep 1</h2></html>",
                "https://myanimelist.net/anime/21/One_Piece/episode/1",
            ),
        ),
    ):
        fetched = await fetcher.fetch_page(page_key)
    assert fetched.is_not_found is False
    assert len(fetched.responses) == 1
    assert fetched.responses[0].request_name == "main"


@pytest.mark.asyncio
async def test_mal_fetcher_episode_invalid_address_and_404_handled() -> None:
    fetcher = MalFetcher()
    with pytest.raises(ValueError, match="must be '<anime_id>:<episode_number>'"):
        await fetcher.fetch_page(
            PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="bad-address",
            )
        )

    with (
        patch.object(
            mal_fetcher,
            "browser_session",
            autospec=True,
            return_value=_StubBrowserSession(),
        ),
        patch.object(
            mal_fetcher,
            "_navigate_and_read",
            autospec=True,
            return_value=(
                '<div class="error404">404 Not Found</div>',
                "https://myanimelist.net/anime/21/_/episode/999999",
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
