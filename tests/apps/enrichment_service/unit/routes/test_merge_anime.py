from __future__ import annotations

import json
import threading
from unittest.mock import create_autospec, patch

import grpc
import pytest
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import (
    MergeAnimeRequest,
    MergeAnimeResponse,
)
from enrichment_service.routes.adapter import EnrichmentRoutes
from enrichment_service.runtime import EnrichmentRuntime
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeReference,
    GeneratedText,
    PageKey,
    PageResponse,
    StoredPage,
)


class _AbortCalledError(Exception):
    pass


def _build_routes() -> EnrichmentRoutes:
    return EnrichmentRoutes(runtime=create_autospec(EnrichmentRuntime, instance=True))


def _build_context() -> grpc.aio.ServicerContext:
    context = create_autospec(grpc.aio.ServicerContext, instance=True)
    context.abort.side_effect = _AbortCalledError()
    return context


@pytest.mark.asyncio
async def test_merge_anime_empty_anime_id_aborts_invalid_argument() -> None:
    routes = _build_routes()
    context = _build_context()

    with pytest.raises(_AbortCalledError):
        await routes.MergeAnime(MergeAnimeRequest(anime_id=""), context)

    context.abort.assert_awaited_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "anime_id is required"
    )


@pytest.mark.asyncio
async def test_merge_anime_valid_page_returns_merged_response() -> None:
    routes = _build_routes()
    context = _build_context()
    page = StoredPage(
        page_key=PageKey(
            provider="anilist",
            entity_type=anime_pb2.ENTITY_TYPE_ANIME,
            address="21",
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(
                    {
                        "data": {
                            "Media": {
                                "id": 21,
                                "title": {"romaji": "One Piece"},
                                "format": "TV",
                                "status": "RELEASING",
                                "description": "Provider synopsis",
                                "siteUrl": "https://anilist.co/anime/21",
                            }
                        }
                    }
                ).encode("utf-8"),
            )
        ],
    )
    response = await routes.MergeAnime(
        MergeAnimeRequest(
            anime_id="0195b42a-0000-7000-8000-000000000099",
            source_pages=[page],
        ),
        context,
    )
    assert response.anime.id == "0195b42a-0000-7000-8000-000000000099"
    assert response.anime.title == "One Piece"
    assert response.anime.synopsis == ""
    assert response.provider_synopses_hash


@pytest.mark.asyncio
async def test_merge_anime_errors_map_to_invalid_argument_and_internal() -> None:
    routes = _build_routes()
    context = _build_context()

    for error in (ValueError("bad page payload"), TypeError("bad json shape")):
        context.abort.reset_mock()
        with patch(
            "enrichment_service.routes.merge_anime.merge_anime_pages",
            autospec=True,
            side_effect=error,
        ):
            with pytest.raises(_AbortCalledError):
                await routes.MergeAnime(MergeAnimeRequest(anime_id="work-1"), context)
            context.abort.assert_awaited_with(
                grpc.StatusCode.INVALID_ARGUMENT, str(error)
            )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_anime.merge_anime_pages",
        autospec=True,
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeAnime(MergeAnimeRequest(anime_id="work-1"), context)
        context.abort.assert_awaited_with(grpc.StatusCode.INTERNAL, "MergeAnime failed")


@pytest.mark.asyncio
async def test_merge_anime_executes_off_event_loop_thread() -> None:
    routes = _build_routes()
    context = _build_context()
    loop_thread_id = threading.get_ident()
    observed_thread_ids: list[int] = []

    def _record_thread(request: MergeAnimeRequest) -> MergeAnimeResponse:
        del request
        observed_thread_ids.append(threading.get_ident())
        return MergeAnimeResponse()

    with patch(
        "enrichment_service.routes.merge_anime.merge_anime_pages",
        autospec=True,
        side_effect=_record_thread,
    ):
        await routes.MergeAnime(MergeAnimeRequest(anime_id="work-1"), context)

    assert observed_thread_ids
    assert observed_thread_ids[0] != loop_thread_id


@pytest.mark.asyncio
async def test_merge_anime_grpc_transport_returns_merged_anime() -> None:
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
            anilist_page = StoredPage(
                page_key=PageKey(
                    provider="anilist",
                    entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                    address="21",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "data": {
                                    "Media": {
                                        "id": 21,
                                        "title": {"romaji": "One Piece"},
                                        "format": "TV",
                                        "status": "RELEASING",
                                        "description": "Provider text",
                                        "siteUrl": "https://anilist.co/anime/21",
                                    }
                                }
                            }
                        ).encode("utf-8"),
                    )
                ],
            )
            response = await stub.MergeAnime(
                MergeAnimeRequest(
                    anime_id="0195b42a-0000-7000-8000-000000000777",
                    source_pages=[anilist_page],
                    generated_texts=[
                        GeneratedText(field_path="synopsis", text="AI synopsis")
                    ],
                    known_episodes=[
                        EpisodeReference(
                            episode_id="ep-1", episode_number=1, title="Ep 1"
                        )
                    ],
                )
            )
            assert response.anime.id == "0195b42a-0000-7000-8000-000000000777"
            assert response.anime.synopsis == "AI synopsis"
            assert response.provider_synopses_hash
    finally:
        await server.stop(grace=0)
