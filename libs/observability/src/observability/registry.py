"""Centralized telemetry registry for platform-wide metrics."""

from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Histogram, UpDownCounter

_TRACER = trace.get_tracer("echora.observability")
_METER = metrics.get_meter("echora.observability")


class ObservabilityRegistry:
    """Implementation of TelemetryRegistry using OpenTelemetry.

    All instruments are created once at module import time against the global
    MeterProvider.  If the provider is the no-op default (i.e. setup_metrics()
    has not been called), all record/add calls are silent no-ops — safe to use
    in any code path regardless of whether observability is enabled.
    """

    # --- RPC layer ---
    RPC_REQUESTS: Counter
    RPC_ERRORS: Counter
    RPC_DURATION: Histogram
    INFLIGHT_RPCS: UpDownCounter

    # --- Database layer ---
    DB_QUERY_DURATION: Histogram
    DB_ERRORS: Counter

    # --- Embedding layer ---
    EMBEDDING_DURATION: Histogram

    # --- Search quality ---
    SEARCH_RESULTS_COUNT: Histogram
    SEARCH_EMPTY_RESULTS: Counter

    # --- Enrichment pipeline ---
    PIPELINE_RUNS: Counter
    PIPELINE_DURATION: Histogram

    # --- Image download layer ---
    IMAGE_DOWNLOAD_DURATION: Histogram
    IMAGE_DOWNLOAD_FAILURES: Counter

    def __init__(self) -> None:
        """Create and register all OTel metric instruments.

        Instruments are created against the global MeterProvider. If
        ``setup_metrics()`` has not been called yet, the provider is the
        no-op default and all subsequent ``record``/``add`` calls are
        silent no-ops.
        """
        # RPC metrics (interceptor-level, all services)
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
        self.INFLIGHT_RPCS = _METER.create_up_down_counter(
            "echora_inflight_rpcs",
            description="Number of RPC calls currently being processed",
        )

        # Database metrics (injected into QdrantClient via TelemetryRegistry)
        self.DB_QUERY_DURATION = _METER.create_histogram(
            "echora_db_query_duration_seconds",
            unit="s",
            description="Database query duration in seconds",
        )
        self.DB_ERRORS = _METER.create_counter(
            "echora_db_errors_total",
            description="Total database errors",
        )

        # Embedding metrics (vector service, most expensive operation)
        self.EMBEDDING_DURATION = _METER.create_histogram(
            "echora_embedding_duration_seconds",
            unit="s",
            description="Embedding model inference duration in seconds",
        )

        # Search quality metrics (vector service business signals)
        self.SEARCH_RESULTS_COUNT = _METER.create_histogram(
            "echora_search_results_count",
            description="Number of results returned per search request",
        )
        self.SEARCH_EMPTY_RESULTS = _METER.create_counter(
            "echora_search_empty_results_total",
            description="Total search requests that returned zero results",
        )

        # Enrichment pipeline metrics
        self.PIPELINE_RUNS = _METER.create_counter(
            "echora_pipeline_runs_total",
            description="Total enrichment pipeline executions",
        )
        self.PIPELINE_DURATION = _METER.create_histogram(
            "echora_pipeline_duration_seconds",
            unit="s",
            description="Enrichment pipeline end-to-end execution duration in seconds",
        )

        # Image download metrics (vision processor, network layer)
        self.IMAGE_DOWNLOAD_DURATION = _METER.create_histogram(
            "echora_image_download_duration_seconds",
            unit="s",
            description="Image download and cache duration per URL",
        )
        self.IMAGE_DOWNLOAD_FAILURES = _METER.create_counter(
            "echora_image_download_failures_total",
            description="Total image download failures after all retries",
        )


# Global singleton — safe to import anywhere; instruments are no-ops until
# setup_metrics() installs a real MeterProvider.
registry = ObservabilityRegistry()
