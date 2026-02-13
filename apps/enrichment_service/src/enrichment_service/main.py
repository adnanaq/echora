"""Start and run the enrichment_service gRPC process.

This module owns application bootstrap for enrichment_service:
it configures logging, builds runtime dependencies, registers the gRPC
service and health checks, and starts the server loop.
"""

from __future__ import annotations

import asyncio
import logging
import os

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from common.config import get_settings

from enrichment_proto.v1 import enrichment_service_pb2_grpc

from .routes import EnrichmentRoutes
from .runtime import build_runtime

settings = get_settings()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=getattr(logging, settings.service.log_level),
    format=settings.service.log_format,
)


async def serve() -> None:
    """Start the enrichment_service gRPC server.

    This function builds runtime dependencies, registers gRPC handlers and
    health checks, and blocks until termination.
    """
    runtime = await build_runtime(settings)
    server = grpc.aio.server()

    servicer = EnrichmentRoutes(runtime=runtime)
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(servicer, server)

    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set(
        "enrichment_service.v1.EnrichmentService",
        health_pb2.HealthCheckResponse.SERVING,
    )

    host = os.getenv("ENRICHMENT_SERVICE_HOST", settings.service.vector_service_host)
    port = int(os.getenv("ENRICHMENT_SERVICE_PORT", "8010"))
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    logger.info("Starting enrichment_service gRPC server on %s", bind)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
