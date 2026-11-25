"""gRPC servicer wiring for enrichment_service routes."""

from __future__ import annotations

import grpc
from enrichment_proto.v1 import enrichment_service_pb2, enrichment_service_pb2_grpc

from ..runtime import EnrichmentRuntime
from . import fetch_page, health, merge_anime, merge_character, merge_episode, pipeline


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
        """Handle FetchPage RPC by delegating to fetch_page route logic.

        Args:
            request: Incoming FetchPageRequest carrying page_key and max_age.
            context: Active gRPC servicer context.

        Returns:
            FetchPageResponse holding the fetched StoredPage.
        """
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

    async def MergeEpisode(
        self,
        request: enrichment_service_pb2.MergeEpisodeRequest,
        context: grpc.aio.ServicerContext,
    ) -> enrichment_service_pb2.MergeEpisodeResponse:
        """Handle MergeEpisode RPC by delegating to merge_episode route logic.

        Args:
            request: Incoming MergeEpisodeRequest message.
            context: Active gRPC servicer context.

        Returns:
            MergeEpisodeResponse containing the merged episode.
        """
        return await merge_episode.merge_episode(self.runtime, request, context)

    async def MergeCharacter(
        self,
        request: enrichment_service_pb2.MergeCharacterRequest,
        context: grpc.aio.ServicerContext,
    ) -> enrichment_service_pb2.MergeCharacterResponse:
        """Handle MergeCharacter RPC by delegating to merge_character route logic.

        Args:
            request: Incoming MergeCharacterRequest message.
            context: Active gRPC servicer context.

        Returns:
            MergeCharacterResponse containing the merged character.
        """
        return await merge_character.merge_character(self.runtime, request, context)
