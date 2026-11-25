"""Integration tests for EnrichmentService.MergeCharacter with PostgresService."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import grpc
import pytest
from common.utils.id_generation import generate_entity_id
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import (
    FetchPageRequest,
    MergeCharacterRequest,
)
from enrichment_service.routes.adapter import EnrichmentRoutes
from postgres_proto.v1 import postgres_service_pb2_grpc
from postgres_proto.v1.postgres_service_pb2 import (
    LoadCharacterRequest,
    SaveCharacterRequest,
    StorePageRequest,
)
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, StoredPage

if TYPE_CHECKING:
    from enrichment_service.runtime import EnrichmentRuntime

pytestmark = pytest.mark.integration

_FIVE_CHARACTERS_PER_PROVIDER: list[tuple[str, list[str]]] = [
    ("anilist", ["40", "17", "11", "1", "246"]),
    ("kitsu", ["1301", "1", "2", "3", "4"]),
    ("mal", ["40", "17", "11", "1", "246"]),
    ("anidb", ["474", "1", "2", "3", "4"]),
    (
        "anime_planet",
        [
            "monkey-d-luffy",
            "naruto-uzumaki",
            "edward-elric",
            "spike-spiegel",
            "son-goku",
        ],
    ),
    ("anisearch", ["311", "1", "2", "3", "4"]),
]


def _run_psql(sql: str, variables: dict[str, str] | None = None) -> str:
    """Execute SQL inside the dev PostgreSQL container and return stdout."""
    docker_bin = shutil.which("docker") or "/usr/bin/docker"
    var_flags = "".join(f" -v {key}={val}" for key, val in (variables or {}).items())
    proc = subprocess.run(  # noqa: S603
        [
            docker_bin,
            "exec",
            "-i",
            "echora-dev-postgres",
            "sh",
            "-c",
            'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -At'
            + var_flags,
        ],
        input=sql,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.asyncio
async def test_merge_character_with_postgres_store_page_end_to_end() -> None:
    """Store character pages in PostgresService and merge via MergeCharacter."""
    server = grpc.aio.server()
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        EnrichmentRoutes(runtime=cast("EnrichmentRuntime", SimpleNamespace())),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    now_ts = to_proto_timestamp(datetime.now(UTC))
    page = StoredPage(
        page_key=PageKey(
            provider="anilist",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="40",
        ),
        fetched_at=now_ts,
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(
                    {
                        "data": {
                            "Character": {
                                "id": 40,
                                "name": {
                                    "full": "Monkey D. Luffy",
                                    "native": "モンキー・D・ルフィ",
                                },
                                "description": "Captain of the Straw Hat Pirates.",
                                "siteUrl": "https://anilist.co/character/40",
                                "media": {
                                    "pageInfo": {"hasNextPage": False},
                                    "edges": [
                                        {
                                            "characterRole": "MAIN",
                                            "node": {
                                                "id": 21,
                                                "type": "ANIME",
                                                "title": {"romaji": "One Piece"},
                                            },
                                        }
                                    ],
                                },
                            }
                        }
                    }
                ).encode("utf-8"),
            )
        ],
    )
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

            stored = await postgres_stub.StorePage(StorePageRequest(page=page))
            assert stored.has_changed in (True, False)

            char_id = generate_entity_id()
            merged = await enrichment_stub.MergeCharacter(
                MergeCharacterRequest(character_id=char_id, pages=[page])
            )
            assert merged.character.id == char_id
            assert merged.character.name == "Monkey D. Luffy"
            assert merged.character.animeography[0].title == "One Piece"
            assert (
                merged.character.animeography[0].role == anime_pb2.CHARACTER_ROLE_MAIN
            )
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_merge_character_five_per_provider_end_to_end() -> None:
    """Fetch, store, load, merge, and idempotently save 5 characters per provider."""
    enrichment_target = os.getenv("ENRICHMENT_SERVICE_ADDR", "localhost:8002")
    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
    created_ids: list[str] = []

    try:
        async with (
            grpc.aio.insecure_channel(enrichment_target) as enrichment_channel,
            grpc.aio.insecure_channel(postgres_target) as postgres_channel,
        ):
            enrichment_stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(
                enrichment_channel
            )
            postgres_stub = postgres_service_pb2_grpc.PostgresServiceStub(
                postgres_channel
            )

            for provider, addresses in _FIVE_CHARACTERS_PER_PROVIDER:
                for address in addresses:
                    fetched = await enrichment_stub.FetchPage(
                        FetchPageRequest(
                            page_key=PageKey(
                                provider=provider,
                                entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                                address=address,
                            )
                        ),
                        timeout=30,
                    )
                    assert not fetched.page.is_not_found, (
                        f"{provider}:{address} returned not_found"
                    )
                    assert not fetched.page.is_dead, f"{provider}:{address} is dead"
                    await postgres_stub.StorePage(StorePageRequest(page=fetched.page))

                    char_id = generate_entity_id()
                    created_ids.append(char_id)
                    await asyncio.to_thread(
                        _run_psql,
                        (
                            "INSERT INTO character (id, name) VALUES (:'char_id', 'Test Char');\n"
                            "INSERT INTO character_source (character_id, provider, entity_type, address) "
                            "VALUES (:'char_id', :'provider', 'character', :'address');"
                        ),
                        {
                            "char_id": char_id,
                            "provider": provider,
                            "address": address,
                        },
                    )

                    loaded = await postgres_stub.LoadCharacter(
                        LoadCharacterRequest(character_id=char_id)
                    )
                    assert len(loaded.pages) == 1
                    merged = await enrichment_stub.MergeCharacter(
                        MergeCharacterRequest(
                            character_id=char_id,
                            pages=loaded.pages,
                            corrections=loaded.corrections,
                        )
                    )
                    assert merged.character.id == char_id
                    assert merged.character.name
                    assert merged.character.sources

                    saved_first = await postgres_stub.SaveCharacter(
                        SaveCharacterRequest(
                            character_id=char_id,
                            expected_version=loaded.version,
                            merge_version=1,
                            character=merged.character,
                        )
                    )
                    assert saved_first.has_changed is True
                    assert saved_first.version == loaded.version + 1

                    loaded_again = await postgres_stub.LoadCharacter(
                        LoadCharacterRequest(character_id=char_id)
                    )
                    merged_again = await enrichment_stub.MergeCharacter(
                        MergeCharacterRequest(
                            character_id=char_id,
                            pages=loaded_again.pages,
                            corrections=loaded_again.corrections,
                        )
                    )
                    saved_second = await postgres_stub.SaveCharacter(
                        SaveCharacterRequest(
                            character_id=char_id,
                            expected_version=loaded_again.version,
                            merge_version=1,
                            character=merged_again.character,
                        )
                    )
                    assert saved_second.has_changed is False
                    assert saved_second.version == saved_first.version
    finally:
        for cid in created_ids:
            await asyncio.to_thread(
                _run_psql,
                (
                    "DELETE FROM outbox WHERE entity_id = :'char_id';\n"
                    "DELETE FROM character WHERE id = :'char_id';"
                ),
                {"char_id": cid},
            )
