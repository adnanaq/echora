"""gRPC servicer wiring for enrichment_service routes."""

from __future__ import annotations

import grpc
from enrichment_proto.v1 import enrichment_service_pb2, enrichment_service_pb2_grpc

from ..runtime import EnrichmentRuntime
from . import fetch_page, health, merge_anime, pipeline


class EnrichmentRoutes(enrichment_service_pb2_grpc.EnrichmentServiceServicer):
    """Route-backed EnrichmentService RPC implementation."""

    def __init__(self, runtime: EnrichmentRuntime) -> None:
        """Initialize enrichment route adapter.

        Args:
            runtime: Initialized runtime dependencies.
        """
        self.runtime = runtime

    async def Health(
        self,
        request: enrichment_service_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> enrichment_service_pb2.HealthResponse:
        """Handle Health RPC by delegating to health route logic."""
        return await health.health(self.runtime, request, context)

    async def RunPipeline(
        self,
        request: enrichment_service_pb2.RunPipelineRequest,
        context: grpc.aio.ServicerContext,
    ) -> enrichment_service_pb2.RunPipelineResponse:
        """Handle RunPipeline RPC by delegating to pipeline route logic."""
        return await pipeline.run_pipeline(self.runtime, request, context)

    async def FetchPage(
        self,
        request: enrichment_service_pb2.FetchPageRequest,
        context: grpc.aio.ServicerContext,
    ) -> enrichment_service_pb2.FetchPageResponse:
        """Handle FetchPage RPC by delegating to fetch_page route logic."""
        return await fetch_page.fetch_page(self.runtime, request, context)

    async def MergeAnime(
        self,
        request: enrichment_service_pb2.MergeAnimeRequest,
        context: grpc.aio.ServicerContext,
    ) -> enrichment_service_pb2.MergeAnimeResponse:
        """Handle MergeAnime RPC by delegating to merge_anime route logic.

        Args:
            request: Incoming MergeAnimeRequest message.
            context: Active gRPC servicer context.

        Returns:
            MergeAnimeResponse containing merged anime, episodes, and claims.
        """
        return await merge_anime.merge_anime(self.runtime, request, context)
