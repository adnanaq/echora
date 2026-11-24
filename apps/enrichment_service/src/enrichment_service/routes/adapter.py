"""gRPC servicer wiring for enrichment_service routes."""

from __future__ import annotations

import grpc
from enrichment_proto.v1 import enrichment_service_pb2, enrichment_service_pb2_grpc
from shared_proto.v1 import review_cases_pb2

from ..runtime import EnrichmentRuntime
from . import health, pipeline, seed_cleaning


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

    async def StartSeedCleaning(
        self,
        request: enrichment_service_pb2.StartSeedCleaningRequest,
        context: grpc.aio.ServicerContext,
    ) -> review_cases_pb2.RunState:
        """Handle StartSeedCleaning RPC by delegating to seed_cleaning route logic.

        Args:
            request: gRPC request specifying seed file, data dir, and fetch flags.
            context: Active gRPC servicer context.

        Returns:
            Current ``RunState`` marked as running.
        """
        return await seed_cleaning.start_seed_cleaning(self.runtime, request, context)

    async def GetSeedCleaningStatus(
        self,
        request: enrichment_service_pb2.GetSeedCleaningStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> review_cases_pb2.RunState:
        """Handle GetSeedCleaningStatus RPC by delegating to seed_cleaning route logic.

        Args:
            request: gRPC request specifying the seed-cleaning data directory.
            context: Active gRPC servicer context.

        Returns:
            Current ``RunState`` protobuf message.
        """
        return await seed_cleaning.get_seed_cleaning_status(
            self.runtime, request, context
        )
