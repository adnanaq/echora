"""OpenTelemetry tracing configuration."""

from typing import Any
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, get_aggregated_resources
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

    # Use aggregated resources to include environment detectors (host, process, etc.)
    resource = get_aggregated_resources(
        detectors=[],
        initial_resource=Resource.create(attributes),
    )

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

def create_linked_span(
    name: str,
    link_context: trace.SpanContext,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> trace.Span:
    """Create a new root span that is linked to an existing context.
    
    This is ideal for long-running batch jobs where you want to separate
    the individual item traces from the main batch trace to avoid 
    massive, unmanageable trace trees.
    """
    tracer = trace.get_tracer("echora.linked")
    link = trace.Link(context=link_context)
    return tracer.start_span(name, kind=kind, attributes=attributes, links=[link])
