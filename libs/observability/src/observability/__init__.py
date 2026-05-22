"""Shared observability bootstrap for Echora services."""

import logging as std_logging
import threading

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
    instrument_redis,
)
from .interceptors import AioServerInterceptor
from .logging import setup_logging, stop_logging
from .metrics import setup_metrics
from .registry import registry
from .tracing import create_linked_span, setup_tracing

logger = std_logging.getLogger(__name__)
_telemetry_initialized = False
_telemetry_init_signature: tuple[object, ...] | None = None
# Guards against concurrent initialization if setup_telemetry is ever called from multiple threads.
_telemetry_lock = threading.Lock()

__all__ = [
    "AioServerInterceptor",
    "instrument_aiohttp_client",
    "instrument_grpc_client",
    "instrument_grpc_server",
    "instrument_qdrant_client",
    "instrument_redis",
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
    enable_redis_instrumentation: bool = False,
    metric_export_interval_millis: int = 15000,
    trace_sample_ratio: float = 1.0,
    log_sample_rate: float = 1.0,
    insecure: bool = True,
) -> None:
    """Initialize logging, tracing, metrics, and optional instrumentation.

    Args:
        service_name: Logical service name (e.g. "echora-vector-service").
        version: Service version string for resource attributes.
        environment: Deployment environment (e.g. "production", "local").
        endpoint: OTLP collector gRPC endpoint (e.g. "http://localhost:4317").
        log_level: Root logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        enable_logging: When True, configures structlog with OTel log bridge.
        enable_tracing: When True, configures TracerProvider with OTLP exporter.
        enable_metrics: When True, configures MeterProvider with OTLP exporter.
        enable_grpc_server_instrumentation: Auto-instrument all gRPC server calls.
            Must be called BEFORE grpc.aio.server() is created.
        enable_grpc_client_instrumentation: Auto-instrument gRPC client stubs.
        enable_aiohttp_client_instrumentation: Auto-instrument aiohttp sessions.
        enable_qdrant_client_instrumentation: Auto-instrument Qdrant client.
        metric_export_interval_millis: How often metrics are pushed to the
            collector (default: 15 000 ms). 15 s is the minimum resolution
            required for multi-window burn-rate SLO alerting.
        trace_sample_ratio: Fraction of root spans to record (0.0–1.0).
            Defaults to 1.0 for local development. Set to 0.05–0.10 in
            production to avoid SDK/collector overload at high throughput.
        log_sample_rate: Fraction of INFO/DEBUG log events to emit (0.0–1.0).
            WARN/ERROR/CRITICAL are never sampled out. Defaults to 1.0 (keep
            all). Set to 0.1 in production to reduce log volume by ~10×.
        insecure: Disable TLS on all OTLP gRPC exporters. True for local/dev
            (plain http://); set False for production with an https:// endpoint.
    """
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
        enable_redis_instrumentation,
        metric_export_interval_millis,
        trace_sample_ratio,
        log_sample_rate,
        insecure,
    )
    with _telemetry_lock:
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
                log_sample_rate=log_sample_rate,
                insecure=insecure,
            )

        if enable_tracing:
            setup_tracing(
                service_name=service_name,
                endpoint=endpoint,
                resource_attributes=resource_attributes,
                sample_ratio=trace_sample_ratio,
                insecure=insecure,
            )

        if enable_metrics:
            setup_metrics(
                service_name=service_name,
                endpoint=endpoint,
                resource_attributes=resource_attributes,
                export_interval_millis=metric_export_interval_millis,
                insecure=insecure,
            )

        if enable_grpc_server_instrumentation:
            instrument_grpc_server()
        if enable_grpc_client_instrumentation:
            instrument_grpc_client()
        if enable_aiohttp_client_instrumentation:
            instrument_aiohttp_client()
        if enable_qdrant_client_instrumentation:
            instrument_qdrant_client()
        if enable_redis_instrumentation:
            instrument_redis()

        _telemetry_initialized = True
        _telemetry_init_signature = init_signature
