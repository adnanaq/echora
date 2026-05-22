"""OpenTelemetry metrics configuration."""

import threading
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider

# TraceBasedExemplarFilter records a trace-ID exemplar on a histogram observation
# only when a sampled active span exists in the current context.  This lets
# Grafana jump directly from a P99 latency spike to the exact causal trace in
# Tempo — without attaching exemplars to untraced, un-jumpable requests.
from opentelemetry.sdk.metrics._internal.exemplar import TraceBasedExemplarFilter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource, get_aggregated_resources

# ---------------------------------------------------------------------------
# Histogram bucket boundaries (all values in seconds unless noted otherwise)
# ---------------------------------------------------------------------------

# RPC: 5 ms → 5 s  — fast gRPC calls up to degraded-network scenarios.
_RPC_DURATION_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]

# DB: 1 ms → 2.5 s — fast Qdrant vector queries up to slow index scans.
_DB_DURATION_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]

# Embedding: 10 ms → 10 s — BGE-M3/OpenCLIP can be 100 ms on CPU, up to
# several seconds for large batches or cold-start model loading.
_EMBEDDING_DURATION_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# Pipeline: 1 s → 600 s — enrichment pipelines are long-running (seconds to
# minutes depending on the number of external API calls and AI stages).
_PIPELINE_DURATION_BUCKETS = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]

# Search results: 0 → 100 — _normalize_limit() caps results at 100.
_SEARCH_RESULTS_BUCKETS = [0, 1, 2, 5, 10, 25, 50, 100]

# Image download: 10 ms → 10 s — covers CDN cache hits (~20 ms) to slow
# origin fetches / retries (~5–10 s) for vision processor batch downloads.
_IMAGE_DOWNLOAD_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

# Enrichment API: 50 ms → 60 s — external REST/GraphQL/scraper calls range from
# fast cache hits (~50 ms) to AniDB XML or crawler timeouts (~30–60 s).
_ENRICHMENT_API_DURATION_BUCKETS = [
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    60.0,
]

# Cache operation: 0.5 ms → 1 s — Redis round-trips are typically sub-ms local,
# but network issues or pipeline batches can push into 10–100 ms range.
_CACHE_OP_DURATION_BUCKETS = [
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
]

_METRICS_LOCK = threading.Lock()
_METRICS_CONFIGURED = False

_HISTOGRAM_VIEWS = [
    View(
        instrument_name="echora_rpc_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_RPC_DURATION_BUCKETS
        ),
    ),
    View(
        instrument_name="echora_db_query_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=_DB_DURATION_BUCKETS),
    ),
    View(
        instrument_name="echora_embedding_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_EMBEDDING_DURATION_BUCKETS
        ),
    ),
    View(
        instrument_name="echora_pipeline_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_PIPELINE_DURATION_BUCKETS
        ),
    ),
    View(
        instrument_name="echora_search_results_count",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_SEARCH_RESULTS_BUCKETS
        ),
    ),
    View(
        instrument_name="echora_image_download_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_IMAGE_DOWNLOAD_BUCKETS
        ),
    ),
    View(
        instrument_name="echora_enrichment_api_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_ENRICHMENT_API_DURATION_BUCKETS
        ),
    ),
    View(
        instrument_name="echora_cache_operation_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=_CACHE_OP_DURATION_BUCKETS
        ),
    ),
]


def setup_metrics(
    *,
    service_name: str,
    endpoint: str,
    resource_attributes: dict | None = None,
    export_interval_millis: int = 15000,
    insecure: bool = True,
) -> None:
    """Configures OpenTelemetry metrics with OTLP exporter.

    Args:
        service_name: Name of the service.
        endpoint: OTLP collector endpoint (e.g., "http://localhost:4317").
        resource_attributes: Additional attributes for the resource.
        export_interval_millis: Interval between metric exports.
            Default is 15 000 ms (15 s) — the minimum resolution required for
            multi-window burn-rate SLO alerting. Using 60 s creates up to a
            60 s alert lag, making fast-burn detection unreliable.
        insecure: Disable TLS on the gRPC connection. True for local/dev
            (plain http://); set False for production with an https:// endpoint.
    """
    global _METRICS_CONFIGURED
    with _METRICS_LOCK:
        if _METRICS_CONFIGURED:
            return

        attributes = {"service.name": service_name}
        if resource_attributes:
            attributes.update(resource_attributes)

        resource = get_aggregated_resources(
            detectors=[],
            initial_resource=Resource.create(attributes),
        )

        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=insecure)
        reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=export_interval_millis
        )

        provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
            views=_HISTOGRAM_VIEWS,
            exemplar_filter=TraceBasedExemplarFilter(),
        )
        metrics.set_meter_provider(provider)
        _METRICS_CONFIGURED = True
