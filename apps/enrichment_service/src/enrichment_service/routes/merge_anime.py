"""MergeAnime gRPC route for enrichment_service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc
from enrichment.pipeline.anime_merge import merge_anime_pages
from enrichment_proto.v1 import enrichment_service_pb2

from .merge_runner import execute_merge_rpc

if TYPE_CHECKING:
    from ..runtime import EnrichmentRuntime


async def merge_anime(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.MergeAnimeRequest,
    context: grpc.aio.ServicerContext,
) -> enrichment_service_pb2.MergeAnimeResponse:
    """Handle MergeAnime RPC by mapping stored pages and merging them.

    Args:
        runtime: Enrichment service runtime dependencies.
        request: MergeAnimeRequest carrying work ID, pages, and overrides.
        context: Active gRPC servicer context for status reporting.

    Returns:
        MergeAnimeResponse with the merged Anime, episodes, claims, and hash.
    """
    del runtime
    if not request.anime_id.strip():
        return await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "anime_id is required"
        )
    return await execute_merge_rpc(merge_anime_pages, request, context, "MergeAnime")
