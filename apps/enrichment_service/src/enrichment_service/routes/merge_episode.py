"""MergeEpisode gRPC route for enrichment_service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc
from enrichment.pipeline.episode_merge import merge_episode_pages
from enrichment_proto.v1 import enrichment_service_pb2

from .merge_runner import execute_merge_rpc

if TYPE_CHECKING:
    from ..runtime import EnrichmentRuntime


async def merge_episode(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.MergeEpisodeRequest,
    context: grpc.aio.ServicerContext,
) -> enrichment_service_pb2.MergeEpisodeResponse:
    """Handle MergeEpisode RPC by mapping stored pages and list entries.

    Args:
        runtime: Enrichment service runtime dependencies.
        request: MergeEpisodeRequest carrying IDs, pages, list entries, and overrides.
        context: Active gRPC servicer context for status reporting.

    Returns:
        MergeEpisodeResponse containing the merged Episode message.
    """
    del runtime
    if not request.episode_id.strip():
        return await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "episode_id is required"
        )
    if not request.anime_id.strip():
        return await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "anime_id is required"
        )
    return await execute_merge_rpc(
        merge_episode_pages, request, context, "MergeEpisode"
    )
