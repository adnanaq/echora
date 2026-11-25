"""MergeCharacter gRPC route for enrichment_service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import grpc
from enrichment.pipeline.character_merge import merge_character_pages
from enrichment_proto.v1 import enrichment_service_pb2

if TYPE_CHECKING:
    from ..runtime import EnrichmentRuntime

logger = logging.getLogger(__name__)


async def merge_character(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.MergeCharacterRequest,
    context: grpc.aio.ServicerContext,
) -> enrichment_service_pb2.MergeCharacterResponse:
    """Handle MergeCharacter RPC by mapping stored pages and applying overrides.

    Args:
        runtime: Enrichment service runtime dependencies.
        request: MergeCharacterRequest carrying character ID, pages, and corrections.
        context: Active gRPC servicer context for status reporting.

    Returns:
        MergeCharacterResponse containing the merged Character message.
    """
    del runtime
    if not request.character_id.strip():
        return await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "character_id is required"
        )

    try:
        return merge_character_pages(request)
    except ValueError as exc:
        return await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    except Exception:
        logger.exception("MergeCharacter failed unexpectedly")
        return await context.abort(grpc.StatusCode.INTERNAL, "MergeCharacter failed")
