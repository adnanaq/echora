"""Adapt vector_service gRPC interfaces to route handlers.

This module implements generated gRPC servicer interfaces and delegates each
RPC method to focused handler modules (`admin` and `search`).
"""

from __future__ import annotations

import grpc
from common.config import Settings

from vector_proto.v1 import (
    vector_admin_pb2,
    vector_admin_pb2_grpc,
    vector_search_pb2,
    vector_search_pb2_grpc,
)

from ..runtime import VectorRuntime
from . import admin, search as search_route


class VectorAdminRoutes(vector_admin_pb2_grpc.VectorAdminServiceServicer):
    """Route-backed VectorAdminService RPC implementation."""

    def __init__(self, runtime: VectorRuntime, settings: Settings) -> None:
        """Initialize admin route adapter.

        Args:
            runtime: Initialized runtime dependencies.
            settings: Active application settings.
        """
        self.runtime = runtime
        self.settings = settings

    async def Health(
        self,
        request: vector_admin_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> vector_admin_pb2.HealthResponse:
        """Handle Health RPC by delegating to admin route logic."""
        return await admin.health(self.runtime, self.settings, request, context)

    async def GetStats(
        self,
        request: vector_admin_pb2.GetStatsRequest,
        context: grpc.aio.ServicerContext,
    ) -> vector_admin_pb2.GetStatsResponse:
        """Handle the single admin stats RPC for vector_service."""
        return await admin.get_stats(self.runtime, request, context)


class VectorSearchRoutes(vector_search_pb2_grpc.VectorSearchServiceServicer):
    """Route-backed VectorSearchService RPC implementation."""

    def __init__(self, runtime: VectorRuntime) -> None:
        """Initialize search route adapter.

        Args:
            runtime: Initialized runtime dependencies.
        """
        self.runtime = runtime

    async def Search(
        self,
        request: vector_search_pb2.SearchRequest,
        context: grpc.aio.ServicerContext,
    ) -> vector_search_pb2.SearchResponse:
        """Handle Search RPC by delegating to search route logic."""
        return await search_route.search(self.runtime, request, context)
