"""MergeAnime gRPC route for enrichment_service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import grpc
from enrichment.pipeline.anime_merge import merge_anime_pages
from enrichment_proto.v1 import enrichment_service_pb2

if TYPE_CHECKING:
    from ..runtime import EnrichmentRuntime

logger = logging.getLogger(__name__)


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
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "anime_id is required")

    try:
        return merge_anime_pages(request)
    except ValueError as exc:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    except Exception:
        logger.exception("MergeAnime failed unexpectedly")
        return await context.abort(grpc.StatusCode.INTERNAL, "MergeAnime failed")
