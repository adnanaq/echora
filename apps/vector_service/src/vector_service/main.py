"""Start and run the vector_service gRPC process.

This module owns application bootstrap for vector_service:
it configures logging, builds runtime dependencies, registers gRPC services,
registers health checks, and starts the server loop.
"""

from __future__ import annotations

import asyncio
import logging

import grpc
from common.config import get_settings
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from vector_proto.v1 import vector_admin_pb2_grpc, vector_search_pb2_grpc

from .routes import VectorAdminRoutes, VectorSearchRoutes
from .runtime import build_runtime

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Start the vector_service gRPC server.

    This function builds runtime dependencies, registers admin/search gRPC
    services and health checks, and blocks until termination.
    """
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.service.log_level),
        format=settings.service.log_format,
    )

    runtime = await build_runtime(settings)
    server = grpc.aio.server()

    admin_servicer = VectorAdminRoutes(runtime=runtime, settings=settings)
    search_servicer = VectorSearchRoutes(runtime=runtime)
    vector_admin_pb2_grpc.add_VectorAdminServiceServicer_to_server(
        admin_servicer, server
    )
    vector_search_pb2_grpc.add_VectorSearchServiceServicer_to_server(
        search_servicer, server
    )

    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set(
        "vector_service.v1.VectorAdminService",
        health_pb2.HealthCheckResponse.SERVING,
    )
    await health_servicer.set(
        "vector_service.v1.VectorSearchService",
        health_pb2.HealthCheckResponse.SERVING,
    )

    bind = (
        f"{settings.service.vector_service_host}:{settings.service.vector_service_port}"
    )
    server.add_insecure_port(bind)
    logger.info("Starting vector_service gRPC server on %s", bind)

    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        logger.info("Shutting down vector_service gRPC server")
        await server.stop(grace=5)
        await runtime.async_qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(serve())
