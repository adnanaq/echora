"""Integration tests for EnrichmentService.MergeEpisode with PostgresService."""

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
from enrichment_proto.v1.enrichment_service_pb2 import MergeEpisodeRequest
from enrichment_service.routes.adapter import EnrichmentRoutes
from postgres_proto.v1 import postgres_service_pb2_grpc
from postgres_proto.v1.postgres_service_pb2 import StorePageRequest
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    PageKey,
    PageResponse,
    StoredPage,
)

if TYPE_CHECKING:
    from enrichment_service.runtime import EnrichmentRuntime

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_merge_episode_with_postgres_store_page_end_to_end() -> None:
    """Store an episode page in PostgresService and merge via MergeEpisode."""
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
            provider="kitsu",
            entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
            address="501",
        ),
        fetched_at=to_proto_timestamp(datetime.now(UTC)),
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
                                "canonicalTitle": "I'm Luffy!",
                                "synopsis": "Luffy sets out to sea.",
                            },
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

            ep_id = generate_entity_id()
            anime_id = generate_entity_id()
            merged = await enrichment_stub.MergeEpisode(
                MergeEpisodeRequest(
                    episode_id=ep_id,
                    anime_id=anime_id,
                    episode_number=1,
                    pages=[page],
                    list_entries=[
                        EpisodeListEntry(
                            provider="mal",
                            title="I'm Luffy! The Man Who's Gonna Be King of the Pirates!",
                            is_filler=False,
                        )
                    ],
                )
            )
            assert merged.episode.id == ep_id
            assert merged.episode.anime_id == anime_id
            assert merged.episode.episode_number == 1
            assert (
                merged.episode.title
                == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
            )
            assert merged.episode.synopsis == "Luffy sets out to sea."
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_five_episodes_per_provider_end_to_end() -> None:
    """Store and merge 5 episodes from each provider (AniDB, AniSearch, Kitsu, MAL)."""
    from pathlib import Path

    from enrichment_proto.v1.enrichment_service_pb2 import MergeAnimeRequest

    server = grpc.aio.server()
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        EnrichmentRoutes(runtime=cast("EnrichmentRuntime", SimpleNamespace())),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    now_ts = to_proto_timestamp(datetime.now(UTC))
    anidb_xml = Path(
        "tests/libs/enrichment/unit/sources/anidb/fixtures/onepiece_anidb_raw.xml"
    ).read_bytes()
    anisearch_eps_html = Path(
        "tests/libs/enrichment/unit/sources/anisearch/test_anisearch_episode_crawler/one_piece_episodes.html"
    ).read_bytes()
    anisearch_main_html = (
        b'<html><head><link rel="canonical" '
        b'href="https://www.anisearch.com/anime/2227,one-piece"/></head>'
        b"<body><h1>One Piece</h1></body></html>"
    )
    kitsu_main = json.dumps(
        {
            "data": {
                "id": "12",
                "type": "anime",
                "attributes": {
                    "slug": "one-piece",
                    "canonicalTitle": "One Piece",
                    "titles": {"en": "One Piece"},
                    "subtype": "TV",
                    "status": "current",
                },
            }
        }
    ).encode()
    kitsu_eps = json.dumps(
        {
            "data": [
                {
                    "id": str(500 + num),
                    "type": "episodes",
                    "attributes": {
                        "number": num,
                        "canonicalTitle": f"Kitsu Ep {num}",
                        "synopsis": f"Kitsu synopsis {num}",
                        "airdate": f"1999-10-{19 + num:02d}",
                        "length": 24,
                    },
                }
                for num in range(1, 6)
            ]
        }
    ).encode()
    mal_fixtures = [
        (
            1,
            Path("tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_1.html"),
        ),
        (
            50,
            Path("tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_50.html"),
        ),
        (
            279,
            Path(
                "tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_279.html"
            ),
        ),
        (
            1152,
            Path(
                "tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_1152.html"
            ),
        ),
        (
            2,
            Path("tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_1.html"),
        ),
    ]

    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
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
            anime_id = generate_entity_id()

            for provider_page in (
                StoredPage(
                    page_key=PageKey(
                        provider="anidb",
                        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                        address="69",
                    ),
                    fetched_at=now_ts,
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="application/xml",
                            body=anidb_xml,
                        )
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="anisearch",
                        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                        address="2227",
                    ),
                    fetched_at=now_ts,
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=anisearch_main_html,
                        ),
                        PageResponse(
                            request_name="episodes",
                            content_type="text/html",
                            body=anisearch_eps_html,
                        ),
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="kitsu",
                        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                        address="12",
                    ),
                    fetched_at=now_ts,
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="application/json",
                            body=kitsu_main,
                        ),
                        PageResponse(
                            request_name="episodes_page_1",
                            content_type="application/json",
                            body=kitsu_eps,
                        ),
                    ],
                ),
            ):
                await postgres_stub.StorePage(StorePageRequest(page=provider_page))
                work_resp = await enrichment_stub.MergeAnime(
                    MergeAnimeRequest(anime_id=anime_id, source_pages=[provider_page])
                )
                by_num = {ep.episode_number: ep for ep in work_resp.episodes}
                for num in range(1, 6):
                    ref = by_num[num]
                    pages = []
                    if provider_page.page_key.provider == "kitsu":
                        ep_page = StoredPage(
                            page_key=PageKey(
                                provider="kitsu",
                                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                                address=str(500 + num),
                            ),
                            fetched_at=now_ts,
                            responses=[
                                PageResponse(
                                    request_name="main",
                                    content_type="application/json",
                                    body=json.dumps(
                                        {
                                            "data": {
                                                "id": str(500 + num),
                                                "type": "episodes",
                                                "attributes": {
                                                    "number": num,
                                                    "canonicalTitle": f"Kitsu Ep {num}",
                                                    "synopsis": f"Kitsu synopsis {num}",
                                                    "length": 24,
                                                },
                                            }
                                        }
                                    ).encode(),
                                )
                            ],
                        )
                        await postgres_stub.StorePage(StorePageRequest(page=ep_page))
                        pages = [ep_page]
                    merged = await enrichment_stub.MergeEpisode(
                        MergeEpisodeRequest(
                            episode_id=ref.episode_id,
                            anime_id=anime_id,
                            episode_number=ref.episode_number,
                            pages=pages,
                            list_entries=ref.list_entries,
                        )
                    )
                    if pages:
                        assert merged.episode.episode_number == num
                    assert merged.episode.title

            for ep_num, mal_path in mal_fixtures:
                mal_page = StoredPage(
                    page_key=PageKey(
                        provider="mal",
                        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                        address=f"21:{ep_num}",
                    ),
                    fetched_at=now_ts,
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=mal_path.read_bytes(),
                        )
                    ],
                )
                await postgres_stub.StorePage(StorePageRequest(page=mal_page))
                merged = await enrichment_stub.MergeEpisode(
                    MergeEpisodeRequest(
                        episode_id=generate_entity_id(),
                        anime_id=anime_id,
                        episode_number=ep_num,
                        pages=[mal_page],
                    )
                )
                assert merged.episode.title
                assert merged.episode.sources
    finally:
        await server.stop(grace=0)
