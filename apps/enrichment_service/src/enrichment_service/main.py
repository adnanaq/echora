"""Start and run the enrichment_service gRPC process.

This module owns application bootstrap for enrichment_service:
it configures logging, builds runtime dependencies, registers the gRPC
service and health checks, and starts the server loop.
"""

from __future__ import annotations

import asyncio
import logging
import signal

import grpc
from common.config import get_settings
from enrichment_proto.v1 import enrichment_service_pb2_grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .routes import EnrichmentRoutes
from .runtime import build_runtime

logger = logging.getLogger(__name__)


def _register_sigterm_shutdown(
    server: grpc.aio.Server,
    *,
    grace: int = 5,
    loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """Register a SIGTERM handler that initiates graceful shutdown.

    Args:
        server: Running gRPC server instance to stop when SIGTERM is received.
        grace: Grace period in seconds passed to ``server.stop``.
        loop: Event loop used to register the signal handler. When omitted, the
            current running loop is used.
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    _shutdown_task: asyncio.Task[None] | None = None

    def _handle_sigterm() -> None:
        nonlocal _shutdown_task
        if _shutdown_task is None or _shutdown_task.done():
            _shutdown_task = asyncio.create_task(server.stop(grace=grace))

    try:
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    except NotImplementedError:
        logger.warning("SIGTERM handler is not supported on this platform")
    except RuntimeError:
        logger.warning("Unable to register SIGTERM handler without a running loop")


async def serve() -> None:
    """Start the enrichment_service gRPC server.

    This function builds runtime dependencies, registers gRPC handlers and
    health checks, and blocks until termination.

    Returns:
        None. This coroutine runs until the server receives termination.
    """
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.service.log_level),
        format=settings.service.log_format,
    )

    runtime = await build_runtime(settings)
    server = grpc.aio.server()

    servicer = EnrichmentRoutes(runtime=runtime)
    enrichment_service_pb2_grpc.add_EnrichmentServiceServicer_to_server(
        servicer, server
    )

    health_servicer = health.aio.HealthServicer()  # ty: ignore[unresolved-attribute]
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set(
        "enrichment_service.v1.EnrichmentService",
        health_pb2.HealthCheckResponse.SERVING,
    )

    host = settings.service.enrichment_service_host
    port = settings.service.enrichment_service_port
    bind = f"{host}:{port}"
    server.add_insecure_port(bind)
    logger.info("Starting enrichment_service gRPC server on %s", bind)

    try:
        await server.start()
        _register_sigterm_shutdown(server, grace=5)
        await server.wait_for_termination()
    finally:
        logger.info("Shutting down enrichment_service gRPC server")
        await server.stop(grace=5)


if __name__ == "__main__":
    asyncio.run(serve())
