"""OpenTelemetry tracing configuration."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(
    *,
    service_name: str,
    endpoint: str,
    resource_attributes: dict | None = None,
) -> None:
    """Configures OpenTelemetry tracing with OTLP exporter.

    Args:
        service_name: Name of the service.
        endpoint: OTLP collector endpoint (e.g., "http://localhost:4317").
        resource_attributes: Additional attributes for the resource.
    """
    attributes = {"service.name": service_name}
    if resource_attributes:
        attributes.update(resource_attributes)

    resource = Resource.create(attributes)

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
