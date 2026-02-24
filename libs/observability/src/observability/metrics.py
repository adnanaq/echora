"""OpenTelemetry metrics configuration."""

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


def setup_metrics(
    *,
    service_name: str,
    endpoint: str,
    resource_attributes: dict | None = None,
) -> None:
    """Configures OpenTelemetry metrics with OTLP exporter.

    Args:
        service_name: Name of the service.
        endpoint: OTLP collector endpoint (e.g., "http://localhost:4317").
        resource_attributes: Additional attributes for the resource.
    """
    attributes = {"service.name": service_name}
    if resource_attributes:
        attributes.update(resource_attributes)

    resource = Resource.create(attributes)

    exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    reader = PeriodicExportingMetricReader(exporter)

    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
