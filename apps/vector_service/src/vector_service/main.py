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
from observability import setup_telemetry
from vector_proto.v1 import vector_admin_pb2_grpc, vector_search_pb2_grpc

from .routes import VectorAdminRoutes, VectorSearchRoutes
from .runtime import VectorRuntime, build_runtime

logger = logging.getLogger(__name__)

_SERVICE_NAMES = (
    "",
    "vector_service.v1.VectorAdminService",
    "vector_service.v1.VectorSearchService",
)


def _setup_observability(settings) -> None:
    if not settings.observability.otel_enabled:
        return

    setup_telemetry(
        service_name="echora-vector-service",
        version=settings.service.api_version,
        environment=settings.environment.value,
        endpoint=settings.observability.otel_exporter_otlp_endpoint,
        log_level=settings.service.log_level,
        enable_logging=settings.observability.otel_enable_logging,
        enable_tracing=settings.observability.otel_enable_tracing,
        enable_metrics=settings.observability.otel_enable_metrics,
        enable_grpc_server_instrumentation=(
            settings.observability.otel_enable_grpc_server_instrumentation
        ),
        enable_grpc_client_instrumentation=(
            settings.observability.otel_enable_grpc_client_instrumentation
        ),
        enable_aiohttp_client_instrumentation=(
            settings.observability.otel_enable_aiohttp_client_instrumentation
        ),
    )


async def _set_health_statuses(
    health_servicer: health.aio.HealthServicer,  # ty: ignore[unresolved-attribute]
    status: int,
) -> None:
    for service_name in _SERVICE_NAMES:
        await health_servicer.set(service_name, status)


async def _publish_initial_readiness(
    runtime: VectorRuntime,
    health_servicer: health.aio.HealthServicer,  # ty: ignore[unresolved-attribute]
) -> None:
    try:
        db_healthy = await runtime.qdrant_client.health_check()
    except Exception:
        logger.exception("Initial Qdrant health check failed")
        db_healthy = False

    status = (
        health_pb2.HealthCheckResponse.SERVING
        if db_healthy
        else health_pb2.HealthCheckResponse.NOT_SERVING
    )
    await _set_health_statuses(health_servicer, status)

    if not db_healthy:
        logger.warning("vector_service started with unhealthy Qdrant dependency")


async def serve() -> None:
    """Start the vector_service gRPC server.

    This function builds runtime dependencies, registers admin/search gRPC
    services and health checks, and blocks until termination.
    """
    settings = get_settings()
    # 1. Initialize Telemetry (MUST BE FIRST)
    _setup_observability(settings)

    runtime = await build_runtime(settings)

    # 2. Configure server with interceptors
    interceptors = []
    if settings.observability.otel_enabled:
        from observability import AioServerInterceptor

        interceptors.append(AioServerInterceptor())

    server = grpc.aio.server(interceptors=interceptors)

    admin_servicer = VectorAdminRoutes(runtime=runtime, settings=settings)
    search_servicer = VectorSearchRoutes(runtime=runtime)
    vector_admin_pb2_grpc.add_VectorAdminServiceServicer_to_server(
        admin_servicer, server
    )
    vector_search_pb2_grpc.add_VectorSearchServiceServicer_to_server(
        search_servicer, server
    )

    health_servicer = health.aio.HealthServicer()  # ty: ignore[unresolved-attribute]
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await _set_health_statuses(
        health_servicer, health_pb2.HealthCheckResponse.NOT_SERVING
    )

    bind = (
        f"{settings.service.vector_service_host}:{settings.service.vector_service_port}"
    )
    server.add_insecure_port(bind)
    logger.info("Starting vector_service gRPC server on %s", bind)

    try:
        await server.start()
        await _publish_initial_readiness(runtime, health_servicer)
        await server.wait_for_termination()
    finally:
        logger.info("Shutting down vector_service gRPC server")
        try:
            await _set_health_statuses(
                health_servicer, health_pb2.HealthCheckResponse.NOT_SERVING
            )
        except Exception:
            logger.exception("Failed to publish NOT_SERVING during shutdown")
        await server.stop(grace=5)
        await runtime.async_qdrant_client.close()
        from observability import stop_logging

        stop_logging()


if __name__ == "__main__":
    asyncio.run(serve())
