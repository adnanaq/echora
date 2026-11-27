"""MergeEpisode gRPC route for enrichment_service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import grpc
from enrichment.pipeline.episode_merge import merge_episode_pages
from enrichment_proto.v1 import enrichment_service_pb2

if TYPE_CHECKING:
    from ..runtime import EnrichmentRuntime

logger = logging.getLogger(__name__)


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

    try:
        return merge_episode_pages(request)
    except ValueError as exc:
        return await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    except Exception:
        logger.exception("MergeEpisode failed unexpectedly")
        return await context.abort(grpc.StatusCode.INTERNAL, "MergeEpisode failed")
