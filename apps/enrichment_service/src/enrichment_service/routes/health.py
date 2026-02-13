"""Health gRPC handler for enrichment_service."""

from __future__ import annotations

import json

import grpc

from enrichment_proto.v1 import enrichment_service_pb2

from ..runtime import EnrichmentRuntime


async def health(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.HealthRequest,
    context: grpc.aio.ServicerContext,
) -> enrichment_service_pb2.HealthResponse:
    """Return enrichment service health status.

    Args:
        runtime: Initialized runtime dependencies.
        request: Health RPC request payload.
        context: gRPC request context.

    Returns:
        Health response with service details.
    """
    del request, context
    return enrichment_service_pb2.HealthResponse(
        healthy=True,
        service="enrichment_service",
        details_json=json.dumps(
            {
                "mode": "sanitization_pipeline",
                "output_dir": runtime.output_dir,
            }
        ),
    )
