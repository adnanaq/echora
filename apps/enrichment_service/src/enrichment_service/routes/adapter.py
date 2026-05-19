"""gRPC servicer wiring for enrichment_service routes."""

from __future__ import annotations

import grpc
from enrichment_proto.v1 import enrichment_service_pb2, enrichment_service_pb2_grpc

from ..runtime import EnrichmentRuntime
from . import health, pipeline


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
