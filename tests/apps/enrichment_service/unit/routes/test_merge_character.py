from __future__ import annotations

import json
import threading
from unittest.mock import create_autospec, patch

import grpc
import pytest
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import (
    MergeCharacterRequest,
    MergeCharacterResponse,
)
from enrichment_service.routes.adapter import EnrichmentRoutes
from enrichment_service.runtime import EnrichmentRuntime
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, StoredPage


class _AbortCalledError(Exception):
    pass


def _build_routes() -> EnrichmentRoutes:
    return EnrichmentRoutes(runtime=create_autospec(EnrichmentRuntime, instance=True))


def _build_context() -> grpc.aio.ServicerContext:
    context = create_autospec(grpc.aio.ServicerContext, instance=True)
    context.abort.side_effect = _AbortCalledError()
    return context


@pytest.mark.asyncio
async def test_merge_character_invalid_id_and_errors_map_to_expected_status() -> None:
    routes = _build_routes()
    context = _build_context()

    with pytest.raises(_AbortCalledError):
        await routes.MergeCharacter(MergeCharacterRequest(character_id=""), context)
    context.abort.assert_awaited_with(
        grpc.StatusCode.INVALID_ARGUMENT, "character_id is required"
    )

    for error in (ValueError("invalid character page"), TypeError("bad payload")):
        context.abort.reset_mock()
        with patch(
            "enrichment_service.routes.merge_character.merge_character_pages",
            autospec=True,
            side_effect=error,
        ):
            with pytest.raises(_AbortCalledError):
                await routes.MergeCharacter(
                    MergeCharacterRequest(character_id="char-1"),
                    context,
                )
            context.abort.assert_awaited_with(
                grpc.StatusCode.INVALID_ARGUMENT, str(error)
            )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_character.merge_character_pages",
        autospec=True,
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeCharacter(
                MergeCharacterRequest(character_id="char-1"),
                context,
            )
        context.abort.assert_awaited_with(
            grpc.StatusCode.INTERNAL, "MergeCharacter failed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "body"),
    [
        ("anilist", b"[]"),
        ("anilist", b'{"data": {"Character": null}}'),
        ("anilist", b'{"data": {"Character": {"id": 1, "name": "oops"}}}'),
        ("kitsu", b'{"data": null}'),
        ("kitsu", b'{"data": {"id": "1", "attributes": "oops"}}'),
    ],
)
async def test_merge_character_malformed_json_payloads_abort_invalid_argument(
    provider: str,
    body: bytes,
) -> None:
    routes = _build_routes()
    context = _build_context()
    request = MergeCharacterRequest(
        character_id="char-bad",
        pages=[
            StoredPage(
                page_key=PageKey(
                    provider=provider,
                    entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                    address="40",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="application/json",
                        body=body,
                    )
                ],
            )
        ],
    )
    with pytest.raises(_AbortCalledError):
        await routes.MergeCharacter(request, context)
    assert context.abort.await_args is not None
    assert context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_merge_character_executes_off_event_loop_thread() -> None:
    routes = _build_routes()
    context = _build_context()
    loop_thread_id = threading.get_ident()
    observed_thread_ids: list[int] = []

    def _record_thread(request: MergeCharacterRequest) -> MergeCharacterResponse:
        del request
        observed_thread_ids.append(threading.get_ident())
        return MergeCharacterResponse()

    with patch(
        "enrichment_service.routes.merge_character.merge_character_pages",
        autospec=True,
        side_effect=_record_thread,
    ):
        await routes.MergeCharacter(
            MergeCharacterRequest(character_id="char-1"),
            context,
        )

    assert observed_thread_ids
    assert observed_thread_ids[0] != loop_thread_id


@pytest.mark.asyncio
async def test_merge_character_grpc_transport_merges_stored_pages() -> None:
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
            response = await stub.MergeCharacter(
                MergeCharacterRequest(
                    character_id="char-40",
                    pages=[
                        StoredPage(
                            page_key=PageKey(
                                provider="anilist",
                                entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                                address="40",
                            ),
                            responses=[
                                PageResponse(
                                    request_name="main",
                                    content_type="application/json",
                                    body=json.dumps(
                                        {
                                            "data": {
                                                "Character": {
                                                    "id": 40,
                                                    "name": {"full": "Monkey D. Luffy"},
                                                    "siteUrl": "https://anilist.co/character/40",
                                                    "media": {
                                                        "pageInfo": {
                                                            "hasNextPage": False
                                                        },
                                                        "edges": [],
                                                    },
                                                }
                                            }
                                        }
                                    ).encode("utf-8"),
                                )
                            ],
                        )
                    ],
                )
            )
            assert response.character.id == "char-40"
            assert response.character.name == "Monkey D. Luffy"
    finally:
        await server.stop(grace=0)
