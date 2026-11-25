"""Unit tests for EnrichmentService.MergeCharacter gRPC route."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch

import grpc
import pytest
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from enrichment_proto.v1.enrichment_service_pb2 import MergeCharacterRequest
from enrichment_service.routes.adapter import EnrichmentRoutes
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, StoredPage

if TYPE_CHECKING:
    from enrichment_service.runtime import EnrichmentRuntime


class _AbortCalledError(Exception):
    """Sentinel raised when mock context.abort is invoked."""


@pytest.mark.asyncio
async def test_merge_character_route_validation_and_error_mapping() -> None:
    """MergeCharacter validates character_id and maps ValueError/Exception to gRPC status."""
    routes = EnrichmentRoutes(runtime=cast("EnrichmentRuntime", SimpleNamespace()))
    context = AsyncMock()
    context.abort.side_effect = _AbortCalledError()

    with pytest.raises(_AbortCalledError):
        await routes.MergeCharacter(MergeCharacterRequest(character_id=""), context)
    context.abort.assert_awaited_with(
        grpc.StatusCode.INVALID_ARGUMENT, "character_id is required"
    )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_character.merge_character_pages",
        side_effect=ValueError("invalid character page"),
    ):
        with pytest.raises(_AbortCalledError):
            await routes.MergeCharacter(
                MergeCharacterRequest(character_id="char-1"),
                context,
            )
        context.abort.assert_awaited_with(
            grpc.StatusCode.INVALID_ARGUMENT, "invalid character page"
        )

    context.abort.reset_mock()
    with patch(
        "enrichment_service.routes.merge_character.merge_character_pages",
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
async def test_merge_character_end_to_end_over_grpc_transport() -> None:
    """End-to-end gRPC server -> stub call merges stored character pages."""
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
            resp = await stub.MergeCharacter(
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
            assert resp.character.id == "char-40"
            assert resp.character.name == "Monkey D. Luffy"
    finally:
        await server.stop(grace=0)
