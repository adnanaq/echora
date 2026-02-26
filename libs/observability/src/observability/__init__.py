"""Shared observability bootstrap for Echora services."""

import logging as std_logging

from .context import (
    extract_context_from_nats_headers,
    extract_context_from_temporal_headers,
    extract_trace_context,
    inject_context_into_nats_headers,
    inject_context_into_temporal_headers,
    inject_trace_context,
)
from .instrumentation import (
    instrument_aiohttp_client,
    instrument_grpc_client,
    instrument_grpc_server,
    instrument_qdrant_client,
)
from .interceptors import AioServerInterceptor
from .logging import setup_logging, stop_logging
from .metrics import setup_metrics
from .registry import registry
from .tracing import create_linked_span, setup_tracing

logger = std_logging.getLogger(__name__)
_telemetry_initialized = False
_telemetry_init_signature: tuple[object, ...] | None = None

__all__ = [
    "AioServerInterceptor",
    "instrument_aiohttp_client",
    "instrument_grpc_client",
    "instrument_grpc_server",
    "instrument_qdrant_client",
    "extract_context_from_nats_headers",
    "extract_context_from_temporal_headers",
    "extract_trace_context",
    "inject_context_into_nats_headers",
    "inject_context_into_temporal_headers",
    "inject_trace_context",
    "setup_logging",
    "stop_logging",
    "setup_metrics",
    "registry",
    "setup_telemetry",
    "setup_tracing",
    "create_linked_span",
]


def setup_telemetry(
    *,
    service_name: str,
    version: str,
    environment: str,
    endpoint: str,
    log_level: str = "INFO",
    enable_logging: bool = True,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_grpc_server_instrumentation: bool = False,
    enable_grpc_client_instrumentation: bool = False,
    enable_aiohttp_client_instrumentation: bool = False,
    enable_qdrant_client_instrumentation: bool = False,
    metric_export_interval_millis: int = 60000,
) -> None:
    """Initialize logging, tracing, metrics, and optional instrumentation."""
    global _telemetry_initialized, _telemetry_init_signature

    init_signature = (
        service_name,
        version,
        environment,
        endpoint,
        log_level,
        enable_logging,
        enable_tracing,
        enable_metrics,
        enable_grpc_server_instrumentation,
        enable_grpc_client_instrumentation,
        enable_aiohttp_client_instrumentation,
        enable_qdrant_client_instrumentation,
        metric_export_interval_millis,
    )
    if _telemetry_initialized:
        if _telemetry_init_signature != init_signature:
            logger.warning(
                "setup_telemetry called with different config after initialization; "
                "ignoring re-initialization request"
            )
        return

    resource_attributes = {
        "service.version": version,
        "deployment.environment": environment,
    }

    if enable_logging:
        setup_logging(
            level=log_level,
            service_name=service_name,
            environment=environment,
            endpoint=endpoint,
        )

    if enable_tracing:
        setup_tracing(
            service_name=service_name,
            endpoint=endpoint,
            resource_attributes=resource_attributes,
        )

    if enable_metrics:
        setup_metrics(
            service_name=service_name,
            endpoint=endpoint,
            resource_attributes=resource_attributes,
            export_interval_millis=metric_export_interval_millis,
        )

    if enable_grpc_server_instrumentation:
        instrument_grpc_server()
    if enable_grpc_client_instrumentation:
        instrument_grpc_client()
    if enable_aiohttp_client_instrumentation:
        instrument_aiohttp_client()
    if enable_qdrant_client_instrumentation:
        instrument_qdrant_client()

    _telemetry_initialized = True
    _telemetry_init_signature = init_signature
