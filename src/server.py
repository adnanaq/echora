"""
Main gRPC server entrypoint.

Initializes and runs the gRPC server with all registered services.
"""
import grpc
import asyncio
import logging

# Import servicers
from src.agent_service import AgentService
from src.admin_service import AdminService

# Import gRPC stubs
from protos import agent_pb2_grpc
from protos import admin_pb2_grpc

# Import health checking services
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

logger = logging.getLogger(__name__)

async def serve_async():
    """
    Starts the consolidated async gRPC server with all services.
    """
    server = grpc.aio.server()

    # Add application services
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentService(), server)
    admin_pb2_grpc.add_AdminServiceServicer_to_server(AdminService(), server)

    # Add the standard health checking service
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Set initial health status for services
    # An empty string denotes the overall server health.
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set("AgentService", health_pb2.HealthCheckResponse.SERVING)
    await health_servicer.set("AdminService", health_pb2.HealthCheckResponse.SERVING)

    # TODO: Get port from settings
    port = "50051"
    server.add_insecure_port(f'[::]:{port}')

    logger.info(f"gRPC server started on port {port}, serving Agent, Admin, and Health services.")
    await server.start()
    await server.wait_for_termination()

def serve():
    """
    Synchronous wrapper for async server (for backward compatibility).
    """
    asyncio.run(serve_async())
