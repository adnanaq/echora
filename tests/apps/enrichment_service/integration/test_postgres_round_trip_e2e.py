"""Full end-to-end tests across EnrichmentService, PostgresService, and PostgreSQL."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import grpc
import pytest
from common.utils.id_generation import generate_entity_id
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import (
    FetchPageRequest,
    HealthRequest,
    MergeAnimeRequest,
    MergeCharacterRequest,
    MergeEpisodeRequest,
)
from google.protobuf.wrappers_pb2 import StringValue
from postgres_proto.v1 import postgres_service_pb2_grpc
from postgres_proto.v1.postgres_service_pb2 import (
    LoadAnimeRequest,
    LoadCharacterRequest,
    LoadEpisodeRequest,
    MarkEpisodeAiredRequest,
    SaveAnimeRequest,
    SaveCharacterRequest,
    SaveEpisodeRequest,
    StorePageRequest,
)
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    EpisodeReference,
    PageKey,
    PageResponse,
    StoredPage,
)

pytestmark = pytest.mark.integration


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
async def test_merge_anime_postgres_full_round_trip_end_to_end() -> None:
    """Verify FetchPage, StorePage, LoadAnime, MergeAnime, SaveAnime, and locking."""
    enrichment_target = os.getenv("ENRICHMENT_SERVICE_ADDR", "localhost:8002")
    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
    anime_id = generate_entity_id()

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

            fetched = await enrichment_stub.FetchPage(
                FetchPageRequest(
                    page_key=PageKey(
                        provider="anilist",
                        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                        address="21",
                    )
                ),
                timeout=30,
            )
            await postgres_stub.StorePage(StorePageRequest(page=fetched.page))

            await asyncio.to_thread(
                _run_psql,
                (
                    "INSERT INTO anime (id) VALUES (:'anime_id');\n"
                    "INSERT INTO anime_source (anime_id, provider, entity_type, address) "
                    "VALUES (:'anime_id', 'anilist', 'anime', '21');"
                ),
                {"anime_id": anime_id},
            )

            loaded_1 = await postgres_stub.LoadAnime(
                LoadAnimeRequest(anime_id=anime_id)
            )
            merged_1 = await enrichment_stub.MergeAnime(
                MergeAnimeRequest(
                    anime_id=anime_id,
                    source_pages=loaded_1.source_pages,
                    partial_source_pages=loaded_1.partial_source_pages,
                    corrections=loaded_1.corrections,
                    known_episodes=loaded_1.known_episodes,
                )
            )
            assert merged_1.anime.id == anime_id
            assert merged_1.anime.title

            saved_1 = await postgres_stub.SaveAnime(
                SaveAnimeRequest(
                    anime_id=anime_id,
                    expected_version=loaded_1.version,
                    merge_version=1,
                    anime=merged_1.anime,
                    relation_claims=merged_1.relation_claims,
                    episodes=merged_1.episodes,
                )
            )
            assert saved_1.has_changed is True
            assert saved_1.version == loaded_1.version + 1

            loaded_2 = await postgres_stub.LoadAnime(
                LoadAnimeRequest(anime_id=anime_id)
            )
            merged_2 = await enrichment_stub.MergeAnime(
                MergeAnimeRequest(
                    anime_id=anime_id,
                    source_pages=loaded_2.source_pages,
                    partial_source_pages=loaded_2.partial_source_pages,
                    corrections=loaded_2.corrections,
                    known_episodes=loaded_2.known_episodes,
                )
            )
            saved_2 = await postgres_stub.SaveAnime(
                SaveAnimeRequest(
                    anime_id=anime_id,
                    expected_version=loaded_2.version,
                    merge_version=1,
                    anime=merged_2.anime,
                    relation_claims=merged_2.relation_claims,
                    episodes=merged_2.episodes,
                )
            )
            assert saved_2.has_changed is False
            assert saved_2.version == saved_1.version

            corr_hex = (
                StringValue(value="One Piece (Corrected)").SerializeToString().hex()
            )
            await asyncio.to_thread(
                _run_psql,
                (
                    "INSERT INTO anime_field_correction "
                    "(anime_id, field_path, value_message, value, approved_by) "
                    "VALUES (:'anime_id', 'title', 'google.protobuf.StringValue', "
                    "decode(:'corr_hex', 'hex'), 'integration-test');"
                ),
                {"anime_id": anime_id, "corr_hex": corr_hex},
            )
            loaded_3 = await postgres_stub.LoadAnime(
                LoadAnimeRequest(anime_id=anime_id)
            )
            merged_3 = await enrichment_stub.MergeAnime(
                MergeAnimeRequest(
                    anime_id=anime_id,
                    source_pages=loaded_3.source_pages,
                    partial_source_pages=loaded_3.partial_source_pages,
                    corrections=loaded_3.corrections,
                    known_episodes=loaded_3.known_episodes,
                )
            )
            assert merged_3.anime.title == "One Piece (Corrected)"
            saved_3 = await postgres_stub.SaveAnime(
                SaveAnimeRequest(
                    anime_id=anime_id,
                    expected_version=loaded_3.version,
                    merge_version=1,
                    anime=merged_3.anime,
                    relation_claims=merged_3.relation_claims,
                    episodes=merged_3.episodes,
                )
            )
            assert saved_3.has_changed is True
            assert saved_3.version == saved_2.version + 1

            with pytest.raises(grpc.aio.AioRpcError) as stale_exc:
                await postgres_stub.SaveAnime(
                    SaveAnimeRequest(
                        anime_id=anime_id,
                        expected_version=loaded_3.version,
                        merge_version=1,
                        anime=merged_3.anime,
                    )
                )
            assert stale_exc.value.code() == grpc.StatusCode.ABORTED
    finally:
        await asyncio.to_thread(
            _run_psql,
            (
                "DELETE FROM outbox WHERE entity_id = :'anime_id';\n"
                "DELETE FROM anime WHERE id = :'anime_id';"
            ),
            {"anime_id": anime_id},
        )


@pytest.mark.asyncio
async def test_merge_episode_postgres_full_round_trip_end_to_end() -> None:
    """Verify StorePage, LoadEpisode, MergeEpisode, SaveEpisode, MarkEpisodeAired, and locking."""
    enrichment_target = os.getenv("ENRICHMENT_SERVICE_ADDR", "localhost:8002")
    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
    anime_id = generate_entity_id()
    episode_id = generate_entity_id()
    ep_address = f"e2e-{episode_id}"
    now_ts = to_proto_timestamp(datetime.now(UTC))
    ep_page = StoredPage(
        page_key=PageKey(
            provider="kitsu",
            entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
            address=ep_address,
        ),
        fetched_at=now_ts,
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(
                    {
                        "data": {
                            "id": ep_address,
                            "type": "episodes",
                            "attributes": {
                                "number": 1,
                                "canonicalTitle": "I'm Luffy!",
                                "synopsis": "Luffy sets out to sea.",
                                "airdate": "1999-10-20",
                                "length": 24,
                            },
                        }
                    }
                ).encode("utf-8"),
            )
        ],
    )

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

            await postgres_stub.StorePage(StorePageRequest(page=ep_page))
            await asyncio.to_thread(
                _run_psql,
                "INSERT INTO anime (id) VALUES (:'anime_id');",
                {"anime_id": anime_id},
            )
            await postgres_stub.SaveAnime(
                SaveAnimeRequest(
                    anime_id=anime_id,
                    expected_version=1,
                    merge_version=1,
                    anime=anime_pb2.Anime(
                        id=anime_id,
                        title="One Piece",
                        type=anime_pb2.ANIME_TYPE_TV,
                        status=anime_pb2.ANIME_STATUS_ONGOING,
                    ),
                    episodes=[
                        EpisodeReference(
                            episode_id=episode_id,
                            episode_number=1,
                            title="I'm Luffy!",
                            page_keys=[ep_page.page_key],
                            list_entries=[
                                EpisodeListEntry(
                                    provider="mal",
                                    title="I'm Luffy! List Title",
                                    is_filler=False,
                                )
                            ],
                        )
                    ],
                )
            )

            loaded_1 = await postgres_stub.LoadEpisode(
                LoadEpisodeRequest(episode_id=episode_id)
            )
            assert loaded_1.anime_id == anime_id
            assert len(loaded_1.pages) == 1

            merged_1 = await enrichment_stub.MergeEpisode(
                MergeEpisodeRequest(
                    episode_id=episode_id,
                    anime_id=loaded_1.anime_id,
                    pages=loaded_1.pages,
                    corrections=loaded_1.corrections,
                    list_entries=loaded_1.list_entries,
                )
            )
            assert merged_1.episode.id == episode_id
            assert merged_1.episode.title == "I'm Luffy! List Title"

            saved_1 = await postgres_stub.SaveEpisode(
                SaveEpisodeRequest(
                    episode_id=episode_id,
                    expected_version=loaded_1.version,
                    merge_version=1,
                    episode=merged_1.episode,
                )
            )
            assert saved_1.has_changed is True
            assert saved_1.version == loaded_1.version + 1

            loaded_2 = await postgres_stub.LoadEpisode(
                LoadEpisodeRequest(episode_id=episode_id)
            )
            merged_2 = await enrichment_stub.MergeEpisode(
                MergeEpisodeRequest(
                    episode_id=episode_id,
                    anime_id=loaded_2.anime_id,
                    pages=loaded_2.pages,
                    corrections=loaded_2.corrections,
                    list_entries=loaded_2.list_entries,
                )
            )
            saved_2 = await postgres_stub.SaveEpisode(
                SaveEpisodeRequest(
                    episode_id=episode_id,
                    expected_version=loaded_2.version,
                    merge_version=1,
                    episode=merged_2.episode,
                )
            )
            assert saved_2.has_changed is False
            assert saved_2.version == saved_1.version

            aired_1 = await postgres_stub.MarkEpisodeAired(
                MarkEpisodeAiredRequest(episode_id=episode_id)
            )
            aired_2 = await postgres_stub.MarkEpisodeAired(
                MarkEpisodeAiredRequest(episode_id=episode_id)
            )
            assert aired_1.has_notified is True
            assert aired_2.has_notified is False

            corr_hex = (
                StringValue(value="I'm Luffy (Corrected)").SerializeToString().hex()
            )
            await asyncio.to_thread(
                _run_psql,
                (
                    "INSERT INTO episode_field_correction "
                    "(episode_id, field_path, value_message, value, approved_by) "
                    "VALUES (:'episode_id', 'title', 'google.protobuf.StringValue', "
                    "decode(:'corr_hex', 'hex'), 'integration-test');"
                ),
                {"episode_id": episode_id, "corr_hex": corr_hex},
            )
            loaded_3 = await postgres_stub.LoadEpisode(
                LoadEpisodeRequest(episode_id=episode_id)
            )
            merged_3 = await enrichment_stub.MergeEpisode(
                MergeEpisodeRequest(
                    episode_id=episode_id,
                    anime_id=loaded_3.anime_id,
                    pages=loaded_3.pages,
                    corrections=loaded_3.corrections,
                    list_entries=loaded_3.list_entries,
                )
            )
            assert merged_3.episode.title == "I'm Luffy (Corrected)"
            saved_3 = await postgres_stub.SaveEpisode(
                SaveEpisodeRequest(
                    episode_id=episode_id,
                    expected_version=loaded_3.version,
                    merge_version=1,
                    episode=merged_3.episode,
                )
            )
            assert saved_3.has_changed is True
            assert saved_3.version == saved_2.version + 1

            with pytest.raises(grpc.aio.AioRpcError) as stale_exc:
                await postgres_stub.SaveEpisode(
                    SaveEpisodeRequest(
                        episode_id=episode_id,
                        expected_version=loaded_3.version,
                        merge_version=1,
                        episode=merged_3.episode,
                    )
                )
            assert stale_exc.value.code() == grpc.StatusCode.ABORTED
    finally:
        await asyncio.to_thread(
            _run_psql,
            (
                "DELETE FROM outbox WHERE entity_id IN (:'anime_id', :'episode_id');\n"
                "DELETE FROM episode WHERE id = :'episode_id';\n"
                "DELETE FROM anime WHERE id = :'anime_id';\n"
                "DELETE FROM provider_page WHERE provider = 'kitsu' AND entity_type = 'episode' AND address = :'ep_address';"
            ),
            {
                "anime_id": anime_id,
                "episode_id": episode_id,
                "ep_address": ep_address,
            },
        )


@pytest.mark.asyncio
async def test_merge_character_postgres_full_scenarios_end_to_end() -> None:
    """Verify multi-provider merge, corrections, outbox, locking, and error statuses."""
    enrichment_target = os.getenv("ENRICHMENT_SERVICE_ADDR", "localhost:8002")
    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
    char_id = generate_entity_id()
    provider_pages = [
        ("anilist", "1"),
        ("kitsu", "2"),
        ("mal", "1"),
        ("anime_planet", "spike-spiegel"),
    ]

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

            for provider, address in provider_pages:
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
                await postgres_stub.StorePage(StorePageRequest(page=fetched.page))

            await asyncio.to_thread(
                _run_psql,
                "INSERT INTO character (id, name) VALUES (:'char_id', 'Spike Temp');",
                {"char_id": char_id},
            )
            for provider, address in provider_pages:
                await asyncio.to_thread(
                    _run_psql,
                    (
                        "INSERT INTO character_source (character_id, provider, entity_type, address) "
                        "VALUES (:'char_id', :'provider', 'character', :'address');"
                    ),
                    {"char_id": char_id, "provider": provider, "address": address},
                )

            loaded = await postgres_stub.LoadCharacter(
                LoadCharacterRequest(character_id=char_id)
            )
            merged = await enrichment_stub.MergeCharacter(
                MergeCharacterRequest(
                    character_id=char_id,
                    pages=loaded.pages,
                    corrections=loaded.corrections,
                )
            )
            assert merged.character.name == "Spike Spiegel"
            assert len(merged.character.sources) >= 4
            assert len(merged.character.animeography) >= 3

            saved_1 = await postgres_stub.SaveCharacter(
                SaveCharacterRequest(
                    character_id=char_id,
                    expected_version=loaded.version,
                    merge_version=1,
                    character=merged.character,
                )
            )
            assert saved_1.has_changed is True
            assert saved_1.version == 2

            outbox_subjects = await asyncio.to_thread(
                _run_psql,
                "SELECT subject FROM outbox WHERE entity_id = :'char_id' ORDER BY created_at;",
                {"char_id": char_id},
            )
            assert "character.created" in outbox_subjects.splitlines()

            correction_hex = (
                StringValue(value="Spike Spiegel (Corrected)").SerializeToString().hex()
            )
            await asyncio.to_thread(
                _run_psql,
                (
                    "INSERT INTO character_field_correction "
                    "(character_id, field_path, value_message, value, approved_by) "
                    "VALUES (:'char_id', 'name', 'google.protobuf.StringValue', "
                    "decode(:'corr_hex', 'hex'), 'integration-test');"
                ),
                {"char_id": char_id, "corr_hex": correction_hex},
            )
            loaded_with_corr = await postgres_stub.LoadCharacter(
                LoadCharacterRequest(character_id=char_id)
            )
            merged_with_corr = await enrichment_stub.MergeCharacter(
                MergeCharacterRequest(
                    character_id=char_id,
                    pages=loaded_with_corr.pages,
                    corrections=loaded_with_corr.corrections,
                )
            )
            assert merged_with_corr.character.name == "Spike Spiegel (Corrected)"

            saved_2 = await postgres_stub.SaveCharacter(
                SaveCharacterRequest(
                    character_id=char_id,
                    expected_version=loaded_with_corr.version,
                    merge_version=1,
                    character=merged_with_corr.character,
                )
            )
            assert saved_2.has_changed is True
            assert saved_2.version == 3

            with pytest.raises(grpc.aio.AioRpcError) as stale_exc:
                await postgres_stub.SaveCharacter(
                    SaveCharacterRequest(
                        character_id=char_id,
                        expected_version=loaded_with_corr.version,
                        merge_version=1,
                        character=merged_with_corr.character,
                    )
                )
            assert stale_exc.value.code() == grpc.StatusCode.ABORTED

            with pytest.raises(grpc.aio.AioRpcError) as missing_exc:
                await postgres_stub.LoadCharacter(
                    LoadCharacterRequest(character_id=generate_entity_id())
                )
            assert missing_exc.value.code() == grpc.StatusCode.NOT_FOUND

            with pytest.raises(grpc.aio.AioRpcError) as invalid_merge_exc:
                await enrichment_stub.MergeCharacter(
                    MergeCharacterRequest(character_id="", pages=loaded.pages)
                )
            assert invalid_merge_exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await asyncio.to_thread(
            _run_psql,
            (
                "DELETE FROM outbox WHERE entity_id = :'char_id';\n"
                "DELETE FROM character WHERE id = :'char_id';"
            ),
            {"char_id": char_id},
        )


@pytest.mark.asyncio
async def test_enrichment_and_postgres_all_rpc_endpoints_and_errors_e2e() -> None:
    """Exercise Health, StorePage idempotency/update, FetchPage errors, and Load/Mark errors."""
    enrichment_target = os.getenv("ENRICHMENT_SERVICE_ADDR", "localhost:8002")
    postgres_target = os.getenv("POSTGRES_SERVICE_ADDR", "localhost:8004")
    test_addr = str(900_000_000 + int.from_bytes(os.urandom(3), "big"))
    now_dt = datetime.now(UTC)
    page_v1_ts = to_proto_timestamp(now_dt - timedelta(seconds=2))
    page_v2_ts = to_proto_timestamp(now_dt)

    async with (
        grpc.aio.insecure_channel(enrichment_target) as enrichment_channel,
        grpc.aio.insecure_channel(postgres_target) as postgres_channel,
    ):
        enrichment_stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(
            enrichment_channel
        )
        postgres_stub = postgres_service_pb2_grpc.PostgresServiceStub(postgres_channel)

        health_resp = await enrichment_stub.Health(HealthRequest())
        assert health_resp.healthy is True
        assert health_resp.service

        page_v1 = StoredPage(
            page_key=PageKey(
                provider="anilist",
                entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                address=test_addr,
            ),
            fetched_at=page_v1_ts,
            responses=[
                PageResponse(
                    request_name="main",
                    content_type="application/json",
                    body=b'{"v": 1}',
                )
            ],
        )
        page_v2 = StoredPage(
            page_key=page_v1.page_key,
            fetched_at=page_v2_ts,
            responses=[
                PageResponse(
                    request_name="main",
                    content_type="application/json",
                    body=b'{"v": 2}',
                )
            ],
        )
        try:
            stored_1 = await postgres_stub.StorePage(StorePageRequest(page=page_v1))
            stored_repeat = await postgres_stub.StorePage(
                StorePageRequest(page=page_v1)
            )
            stored_updated = await postgres_stub.StorePage(
                StorePageRequest(page=page_v2)
            )
            assert stored_1.has_changed is True
            assert stored_repeat.has_changed is False
            assert stored_updated.has_changed is True

            with pytest.raises(grpc.aio.AioRpcError) as bad_fetch:
                await enrichment_stub.FetchPage(
                    FetchPageRequest(
                        page_key=PageKey(
                            provider="unknown_provider",
                            entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                            address="1",
                        )
                    )
                )
            assert bad_fetch.value.code() == grpc.StatusCode.INVALID_ARGUMENT

            for bad_call in (
                postgres_stub.LoadAnime(LoadAnimeRequest(anime_id="not-a-uuid")),
                postgres_stub.LoadEpisode(LoadEpisodeRequest(episode_id="not-a-uuid")),
                postgres_stub.MarkEpisodeAired(
                    MarkEpisodeAiredRequest(episode_id="not-a-uuid")
                ),
            ):
                with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                    await bad_call
                assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

            for missing_call in (
                postgres_stub.LoadAnime(
                    LoadAnimeRequest(anime_id=generate_entity_id())
                ),
                postgres_stub.LoadEpisode(
                    LoadEpisodeRequest(episode_id=generate_entity_id())
                ),
                postgres_stub.MarkEpisodeAired(
                    MarkEpisodeAiredRequest(episode_id=generate_entity_id())
                ),
            ):
                with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                    await missing_call
                assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
        finally:
            await asyncio.to_thread(
                _run_psql,
                "DELETE FROM provider_page WHERE provider = 'anilist' AND entity_type = 'anime' AND address = :'addr';",
                {"addr": test_addr},
            )
