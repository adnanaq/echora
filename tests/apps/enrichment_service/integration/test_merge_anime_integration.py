"""Integration tests for EnrichmentService.MergeAnime with PostgresService."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import grpc
import pytest
from common.utils.id_generation import generate_entity_id
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import MergeAnimeRequest
from enrichment_service.routes.adapter import EnrichmentRoutes
from postgres_proto.v1 import postgres_service_pb2_grpc
from postgres_proto.v1.postgres_service_pb2 import StorePageRequest
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, StoredPage

if TYPE_CHECKING:
    from enrichment_service.runtime import EnrichmentRuntime

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_merge_anime_with_postgres_store_page_end_to_end() -> None:
    """Store pages in PostgresService and merge them via EnrichmentService.MergeAnime."""
    server = grpc.aio.server()
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        EnrichmentRoutes(runtime=cast("EnrichmentRuntime", SimpleNamespace())),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
    page = StoredPage(
        page_key=PageKey(
            provider="anilist",
            entity_type=anime_pb2.ENTITY_TYPE_ANIME,
            address="21",
        ),
        fetched_at=to_proto_timestamp(datetime.now(UTC)),
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
                                "description": "Provider synopsis.",
                                "siteUrl": "https://anilist.co/anime/21",
                            }
                        }
                    }
                ).encode("utf-8"),
            )
        ],
    )
    try:
        async with (
            grpc.aio.insecure_channel(f"127.0.0.1:{port}") as enrichment_channel,
            grpc.aio.insecure_channel(postgres_target) as postgres_channel,
        ):
            enrichment_stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(
                enrichment_channel
            )
            postgres_stub = postgres_service_pb2_grpc.PostgresServiceStub(
                postgres_channel
            )

            stored = await postgres_stub.StorePage(StorePageRequest(page=page))
            assert stored.has_changed in (True, False)

            work_id = generate_entity_id()
            merged = await enrichment_stub.MergeAnime(
                MergeAnimeRequest(anime_id=work_id, source_pages=[page])
            )
            assert merged.anime.id == work_id
            assert merged.anime.title == "One Piece"
            assert merged.anime.synopsis == ""
            assert list(merged.anime.sources) == ["https://anilist.co/anime/21"]
            assert merged.provider_synopses_hash
    finally:
        await server.stop(grace=0)
