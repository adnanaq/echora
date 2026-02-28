"""OpenTelemetry tracing configuration."""

from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, get_aggregated_resources
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased


def setup_tracing(
    *,
    service_name: str,
    endpoint: str,
    resource_attributes: dict | None = None,
    sample_ratio: float = 1.0,
) -> None:
    """Configures OpenTelemetry tracing with OTLP exporter.

    Args:
        service_name: Name of the service.
        endpoint: OTLP collector endpoint (e.g., "http://localhost:4317").
        resource_attributes: Additional attributes for the resource.
        sample_ratio: Fraction of root spans to sample (0.0–1.0). Defaults to
            1.0 (AlwaysOn) for local development. Set to 0.05–0.1 in production
            to avoid overwhelming the collector at scale. Child spans respect the
            parent decision via ParentBased, so end-to-end traces stay coherent.
    """
    attributes = {"service.name": service_name}
    if resource_attributes:
        attributes.update(resource_attributes)

    # Use aggregated resources to include environment detectors (host, process, etc.)
    resource = get_aggregated_resources(
        detectors=[],
        initial_resource=Resource.create(attributes),
    )

    # ParentBased: honour upstream sampling decisions (e.g. from an API gateway
    # that injected a sampled traceparent header); apply TraceIdRatioBased only
    # to root spans that have no upstream parent.
    sampler = ParentBased(root=TraceIdRatioBased(sample_ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)
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
    """Create a new root span linked to an existing span context.

    Unlike child spans, linked spans start a new trace rather than continuing
    the parent's. This is ideal for long-running batch jobs where individual
    item traces should be separate from the triggering batch trace to avoid
    massive, unmanageable trace trees in Tempo.

    Args:
        name: Name of the new span.
        link_context: SpanContext to link to (e.g. from the batch root span).
        kind: SpanKind for the new span. Defaults to INTERNAL.
        attributes: Optional key/value attributes to set on the span at creation.

    Returns:
        A new, already-started Span. The caller is responsible for ending it
        (e.g. via ``with`` statement or explicit ``span.end()``).
    """
    tracer = trace.get_tracer("echora.linked")
    link = trace.Link(context=link_context)
    return tracer.start_span(name, kind=kind, attributes=attributes, links=[link])
