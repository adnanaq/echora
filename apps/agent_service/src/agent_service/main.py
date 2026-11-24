"""Start and run the agent_service gRPC server with health checks."""

import asyncio
import logging
import signal

import grpc
from agent_proto.v1 import agent_service_pb2, agent_service_pb2_grpc
from agent_service.routes import AgentRoutes
from agent_service.runtime import build_runtime
from common.config import Settings, get_settings
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

logger = logging.getLogger(__name__)


def _register_sigterm_shutdown(server: grpc.aio.Server, *, grace: int = 5) -> None:
    """Register a SIGTERM handler that gracefully stops the gRPC server.

    Args:
        server: Running gRPC server instance to stop on SIGTERM.
        grace: Grace period in seconds passed to ``server.stop``.
    """
    loop = asyncio.get_running_loop()
    shutdown_task: asyncio.Task[None] | None = None

    def handle_sigterm() -> None:
        nonlocal shutdown_task
        if shutdown_task is None or shutdown_task.done():
            shutdown_task = asyncio.create_task(server.stop(grace=grace))

    try:
        loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
    except NotImplementedError, RuntimeError:
        logger.warning("SIGTERM handler could not be registered")


async def build_grpc_server(settings: Settings) -> tuple[grpc.aio.Server, str]:
    """Create the configured gRPC server with AgentService and health checks.

    Args:
        settings: Resolved application settings.

    Returns:
        The unstarted ``grpc.aio.Server`` and its ``host:port`` listen address.
    """
    server = grpc.aio.server()
    runtime = build_runtime(settings)
    agent_service_pb2_grpc.add_AgentServiceServicer_to_server(
        AgentRoutes(runtime),
        server,
    )

    health_servicer = health.aio.HealthServicer()  # ty: ignore[unresolved-attribute]
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    service_name = agent_service_pb2.DESCRIPTOR.services_by_name[
        "AgentService"
    ].full_name
    for target_name in ("", service_name):
        await health_servicer.set(target_name, health_pb2.HealthCheckResponse.SERVING)

    listen_address = (
        f"{settings.service.agent_service_host}:{settings.service.agent_service_port}"
    )
    server.add_insecure_port(listen_address)
    return server, listen_address


async def serve() -> None:
    """Start the agent_service gRPC server and block until termination."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.service.log_level, logging.INFO)
    )
    server, listen_address = await build_grpc_server(settings)
    await server.start()
    _register_sigterm_shutdown(server)
    logger.info("agent_service listening on %s", listen_address)
    await server.wait_for_termination()


def main() -> None:
    """Run the agent_service asyncio entry point."""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
