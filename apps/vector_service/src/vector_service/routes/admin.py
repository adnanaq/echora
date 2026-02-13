"""Handle vector_service admin RPCs.

This module exposes a lightweight `Health` RPC and a single rich `GetStats`
RPC for local admin introspection.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import grpc
from common.config import Settings

from vector_proto.v1 import vector_admin_pb2

from ..runtime import VectorRuntime
from .shared import error

logger = logging.getLogger(__name__)


def _health_database_payload(stats: dict[str, object], db_healthy: bool) -> dict[str, object]:
    """Build compact health payload for fast readiness checks.

    Args:
        stats: Qdrant stats map.
        db_healthy: Qdrant health state.

    Returns:
        Small JSON-serializable payload used by Health.
    """
    return {
        "healthy": db_healthy,
        "collection_name": stats.get("collection_name", "unknown"),
        "document_count": stats.get("total_documents", 0),
    }


def _build_stats_payload(runtime: VectorRuntime, stats: dict[str, object]) -> dict[str, object]:
    """Build rich admin stats payload for diagnostics.

    Args:
        runtime: Initialized service runtime dependencies.
        stats: Raw Qdrant stats map.

    Returns:
        JSON-serializable admin payload including collection and processor metadata.
    """
    return {
        "stats": stats,
        "collection": {
            "collection_name": runtime.qdrant_client.collection_name,
            "vector_size": runtime.qdrant_client.vector_size,
            "image_vector_size": runtime.qdrant_client.image_vector_size,
            "distance_metric": runtime.qdrant_client.distance_metric,
        },
        "processors": {
            "text_processor": runtime.text_processor.get_model_info(),
            "vision_processor": runtime.vision_processor.get_model_info(),
        },
    }


async def health(
    runtime: VectorRuntime,
    settings: Settings,
    request: vector_admin_pb2.HealthRequest,
    context: grpc.aio.ServicerContext,
) -> vector_admin_pb2.HealthResponse:
    """Return vector service and Qdrant health status.

    Args:
        runtime: Initialized service runtime dependencies.
        settings: Active application settings.
        request: Health RPC request payload.
        context: gRPC request context.

    Returns:
        Health state and compact database payload for readiness checks.
    """
    del request, context
    try:
        db_healthy = await runtime.qdrant_client.health_check()
        stats: dict[str, object] = {}
        if db_healthy:
            stats = await runtime.qdrant_client.get_stats()
        return vector_admin_pb2.HealthResponse(
            healthy=db_healthy,
            timestamp=datetime.now(UTC).isoformat(),
            service="vector_service",
            version=settings.service.api_version,
            database_json=json.dumps(
                _health_database_payload(stats, db_healthy)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health RPC failed")
        return vector_admin_pb2.HealthResponse(
            healthy=False,
            timestamp=datetime.now(UTC).isoformat(),
            service="vector_service",
            version=settings.service.api_version,
            error=error(
                code="HEALTH_FAILED",
                message=str(exc),
                retryable=False,
            ),
        )


async def get_stats(
    runtime: VectorRuntime,
    request: vector_admin_pb2.GetStatsRequest,
    context: grpc.aio.ServicerContext,
) -> vector_admin_pb2.GetStatsResponse:
    """Return rich admin stats payload for local diagnostics.

    Args:
        runtime: Initialized service runtime dependencies.
        request: Stats RPC request payload.
        context: gRPC request context.

    Returns:
        JSON payload with collection stats and vector/model metadata.
    """
    del request, context
    try:
        stats = await runtime.qdrant_client.get_stats()
        payload = _build_stats_payload(runtime, stats)
        return vector_admin_pb2.GetStatsResponse(stats_json=json.dumps(payload))
    except Exception as exc:  # noqa: BLE001
        logger.exception("GetStats RPC failed")
        return vector_admin_pb2.GetStatsResponse(
            error=error("GET_STATS_FAILED", str(exc))
        )
