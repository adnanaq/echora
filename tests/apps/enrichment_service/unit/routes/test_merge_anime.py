"""Unit tests for the EnrichmentService MergeAnime gRPC route."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import grpc
import pytest
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import MergeAnimeRequest
from enrichment_service.routes.adapter import EnrichmentRoutes
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeReference,
    GeneratedText,
    PageKey,
    PageResponse,
    StoredPage,
)


class _AbortCalledError(Exception):
    """Sentinel raised when mock context.abort is invoked."""


@pytest.mark.asyncio
async def test_merge_anime_route_requires_anime_id() -> None:
    """MergeAnime aborts with INVALID_ARGUMENT when anime_id is empty."""
    routes = EnrichmentRoutes(runtime=SimpleNamespace())  # type: ignore[arg-type]
    context = AsyncMock()
    context.abort.side_effect = _AbortCalledError()

    with pytest.raises(_AbortCalledError):
        await routes.MergeAnime(MergeAnimeRequest(anime_id=""), context)

    context.abort.assert_awaited_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "anime_id is required"
    )


@pytest.mark.asyncio
async def test_merge_anime_route_returns_merged_response() -> None:
    """MergeAnime delegates to merge_anime_pages and returns MergeAnimeResponse."""
    routes = EnrichmentRoutes(runtime=SimpleNamespace())  # type: ignore[arg-type]
    context = AsyncMock()
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
    resp = await routes.MergeAnime(
        MergeAnimeRequest(
            anime_id="0195b42a-0000-7000-8000-000000000099",
            source_pages=[page],
        ),
        context,
    )
    assert resp.anime.id == "0195b42a-0000-7000-8000-000000000099"
    assert resp.anime.title == "One Piece"
    assert resp.anime.synopsis == ""
    assert resp.provider_synopses_hash


@pytest.mark.asyncio
async def test_merge_anime_route_maps_value_and_internal_errors() -> None:
    """MergeAnime maps ValueError to INVALID_ARGUMENT and other errors to INTERNAL."""
    routes = EnrichmentRoutes(runtime=SimpleNamespace())  # type: ignore[arg-type]
    context = AsyncMock()
    context.abort.side_effect = _AbortCalledError()

    with patch(
        "enrichment_service.routes.merge_anime.merge_anime_pages",
        side_effect=ValueError("bad page payload"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeAnime(MergeAnimeRequest(anime_id="work-1"), context)
        context.abort.assert_awaited_with(
            grpc.StatusCode.INVALID_ARGUMENT, "bad page payload"
        )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_anime.merge_anime_pages",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeAnime(MergeAnimeRequest(anime_id="work-1"), context)
        context.abort.assert_awaited_with(grpc.StatusCode.INTERNAL, "MergeAnime failed")


@pytest.mark.asyncio
async def test_merge_anime_end_to_end_over_grpc_transport() -> None:
    """End-to-end gRPC server -> stub call merges pages, episodes, claims, and synopsis."""
    server = grpc.aio.server()
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        EnrichmentRoutes(runtime=SimpleNamespace()),  # type: ignore[arg-type]
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
