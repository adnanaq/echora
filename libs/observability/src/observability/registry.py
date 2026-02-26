"""Centralized telemetry registry for platform-wide metrics."""

from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Histogram
from vector_db_interface import TelemetryRegistry

_TRACER = trace.get_tracer("echora.observability")
_METER = metrics.get_meter("echora.observability")

class ObservabilityRegistry(TelemetryRegistry):
    """Implementation of TelemetryRegistry using OpenTelemetry."""
    
    # Explicitly satisfy the Protocol with OTel types
    RPC_REQUESTS: Counter
    RPC_ERRORS: Counter
    RPC_DURATION: Histogram
    DB_QUERY_DURATION: Histogram
    DB_ERRORS: Counter
    
    def __init__(self):
        # RPC Metrics
        self.RPC_REQUESTS = _METER.create_counter(
            "echora_rpc_requests_total",
            description="Total RPC requests handled by the service",
        )
        self.RPC_ERRORS = _METER.create_counter(
            "echora_rpc_errors_total",
            description="Total RPC requests that ended in error",
        )
        self.RPC_DURATION = _METER.create_histogram(
            "echora_rpc_duration_seconds",
            unit="s",
            description="RPC handler duration in seconds",
        )

        # Database Metrics
        self.DB_QUERY_DURATION = _METER.create_histogram(
            "echora_db_query_duration_seconds",
            unit="s",
            description="Database query duration in seconds",
        )
        self.DB_ERRORS = _METER.create_counter(
            "echora_db_errors_total",
            description="Total database errors",
        )

# Global instance
registry = ObservabilityRegistry()
