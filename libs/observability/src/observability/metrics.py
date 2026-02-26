"""OpenTelemetry metrics configuration."""

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource, get_aggregated_resources

# Bucket boundaries in seconds, tuned for sub-second gRPC and DB latencies.
# RPC: 5ms → 5s covers fast local calls up to degraded network scenarios.
_RPC_DURATION_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
# DB: 1ms → 2.5s covers fast Qdrant vector queries up to slow index scans.
_DB_DURATION_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]

_HISTOGRAM_VIEWS = [
    View(
        instrument_name="echora_rpc_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=_RPC_DURATION_BUCKETS),
    ),
    View(
        instrument_name="echora_db_query_duration_seconds",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=_DB_DURATION_BUCKETS),
    ),
]


def setup_metrics(
    *,
    service_name: str,
    endpoint: str,
    resource_attributes: dict | None = None,
    export_interval_millis: int = 60000,
) -> None:
    """Configures OpenTelemetry metrics with OTLP exporter.

    Args:
        service_name: Name of the service.
        endpoint: OTLP collector endpoint (e.g., "http://localhost:4317").
        resource_attributes: Additional attributes for the resource.
        export_interval_millis: Interval between metric exports (default: 60s).
    """
    attributes = {"service.name": service_name}
    if resource_attributes:
        attributes.update(resource_attributes)

    resource = get_aggregated_resources(
        detectors=[],
        initial_resource=Resource.create(attributes),
    )

    exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=export_interval_millis
    )

    provider = MeterProvider(resource=resource, metric_readers=[reader], views=_HISTOGRAM_VIEWS)
    metrics.set_meter_provider(provider)
