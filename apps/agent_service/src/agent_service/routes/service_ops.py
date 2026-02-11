"""Operational gRPC servicer for runtime diagnostics and metadata.

This module provides internal-only RPCs for health, service information,
database stats, and collection metadata.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import grpc
from google.protobuf.empty_pb2 import Empty

from agent.v1 import service_ops_pb2, service_ops_pb2_grpc

from ..main import AgentService
from ..utils.proto_utils import struct_from_dict

logger = logging.getLogger(__name__)


class ServiceOps(service_ops_pb2_grpc.ServiceOpsServicer):
    """gRPC servicer exposing vector/runtime operational diagnostics."""

    def __init__(self, runtime: AgentService) -> None:
        """Initializes the operations gRPC servicer.

        Args:
            runtime: Shared runtime dependency container.
        """
        self._rt = runtime

    async def GetServiceInfo(
        self,
        request: Empty,
        context: grpc.aio.ServicerContext,
    ) -> service_ops_pb2.GetServiceInfoResponse:
        """Returns service metadata and available internal RPC endpoints.

        Args:
            request: Empty protobuf request.
            context: gRPC servicer context.

        Returns:
            Service metadata and endpoint map.
        """
        del request
        del context
        return service_ops_pb2.GetServiceInfoResponse(
            service="anime-agent-service",
            version=self._rt.app_settings.service.api_version,
            description="Internal gRPC service for agentic search and diagnostics.",
            endpoints={
                "search_ai": "/agent.v1.AgentSearchService/SearchAI",
                "service_info": "/agent.v1.ServiceOps/GetServiceInfo",
                "health_status": "/agent.v1.ServiceOps/GetHealthStatus",
                "database_stats": "/agent.v1.ServiceOps/GetDatabaseStats",
                "collection_info": "/agent.v1.ServiceOps/GetCollectionInfo",
                "grpc_health_check": "/grpc.health.v1.Health/Check",
            },
        )

    async def GetHealthStatus(
        self,
        request: Empty,
        context: grpc.aio.ServicerContext,
    ) -> service_ops_pb2.GetHealthStatusResponse:
        """Returns runtime and Qdrant dependency health.

        Args:
            request: Empty protobuf request.
            context: gRPC servicer context.

        Returns:
            Health response with dependency status and collection diagnostics.
        """
        del request
        del context
        db_healthy = await self._rt.qdrant.health_check()
        stats: dict[str, object] = {}
        if db_healthy:
            stats = await self._rt.qdrant.get_stats()
        timestamp = datetime.now(UTC).isoformat()
        return service_ops_pb2.GetHealthStatusResponse(
            status="healthy" if db_healthy else "unhealthy",
            timestamp=timestamp,
            service="anime-agent-service",
            version=self._rt.app_settings.service.api_version,
            database_healthy=db_healthy,
            collection_name=str(
                stats.get("collection_name", self._rt.qdrant.collection_name)
            ),
            document_count=int(stats.get("total_documents", 0)),
        )

    async def GetDatabaseStats(
        self,
        request: Empty,
        context: grpc.aio.ServicerContext,
    ) -> service_ops_pb2.GetDatabaseStatsResponse:
        """Returns collection-level vector database statistics.

        Args:
            request: Empty protobuf request.
            context: gRPC servicer context.

        Returns:
            Collection statistics and additional diagnostics.
        """
        del request
        stats = await self._rt.qdrant.get_stats()
        if "error" in stats:
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Failed to retrieve stats: {stats['error']}",
            )

        return service_ops_pb2.GetDatabaseStatsResponse(
            collection_name=str(
                stats.get("collection_name", self._rt.qdrant.collection_name)
            ),
            total_documents=int(stats.get("total_documents", 0)),
            vector_size=int(stats.get("vector_size", self._rt.qdrant.vector_size)),
            distance_metric=str(
                stats.get("distance_metric", self._rt.qdrant.distance_metric)
            ),
            status=str(stats.get("status", "unknown")),
            additional_stats=struct_from_dict(
                {
                    "optimizer_status": stats.get("optimizer_status"),
                    "indexed_vectors_count": stats.get("indexed_vectors_count"),
                    "points_count": stats.get("points_count"),
                }
            ),
        )

    async def GetCollectionInfo(
        self,
        request: Empty,
        context: grpc.aio.ServicerContext,
    ) -> service_ops_pb2.GetCollectionInfoResponse:
        """Returns detailed collection configuration plus processor metadata.

        Args:
            request: Empty protobuf request.
            context: gRPC servicer context.

        Returns:
            Collection configuration, stats, and embedding processor info.
        """
        del request
        stats = await self._rt.qdrant.get_stats()
        if "error" in stats:
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Failed to retrieve collection information: {stats['error']}",
            )

        processors = {
            "text_processor": self._rt.text_processor.get_model_info(),
            "vision_processor": self._rt.vision_processor.get_model_info(),
        }
        return service_ops_pb2.GetCollectionInfoResponse(
            collection_name=self._rt.qdrant.collection_name,
            vector_size=self._rt.qdrant.vector_size,
            image_vector_size=self._rt.qdrant.image_vector_size,
            distance_metric=self._rt.qdrant.distance_metric,
            stats=struct_from_dict(stats),
            processors=struct_from_dict(processors),
        )
