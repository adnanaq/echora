from __future__ import annotations

from typing import Any, cast

from observability.context import (
    extract_trace_context,
    inject_context_into_nats_headers,
    inject_context_into_temporal_headers,
    inject_trace_context,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_TRACER: trace.Tracer | None = None


def _setup_tracer() -> trace.Tracer:
    global _TRACER
    if _TRACER is not None:
        return _TRACER

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer(__name__)
    return _TRACER


def test_trace_context_round_trip_preserves_parent_child_relationship() -> None:
    tracer = _setup_tracer()

    with tracer.start_as_current_span("parent") as parent_span:
        headers = inject_trace_context({})

    extracted_context = extract_trace_context(headers)
    with tracer.start_as_current_span("child", context=extracted_context) as child_span:
        child_context = child_span.get_span_context()
        parent_context = parent_span.get_span_context()
        parent = cast(Any, child_span).parent

        assert child_context.trace_id == parent_context.trace_id
        assert parent is not None
        assert parent.span_id == parent_context.span_id


def test_extract_trace_context_handles_missing_headers() -> None:
    tracer = _setup_tracer()
    extracted_context = extract_trace_context({})

    with tracer.start_as_current_span("root", context=extracted_context) as span:
        assert cast(Any, span).parent is None


def test_extract_trace_context_handles_invalid_traceparent() -> None:
    tracer = _setup_tracer()
    extracted_context = extract_trace_context({"traceparent": "invalid"})

    with tracer.start_as_current_span("root", context=extracted_context) as span:
        assert cast(Any, span).parent is None


def test_nats_and_temporal_helpers_inject_traceparent() -> None:
    tracer = _setup_tracer()

    with tracer.start_as_current_span("producer"):
        nats_headers = inject_context_into_nats_headers({})
        temporal_headers = inject_context_into_temporal_headers({})

    assert "traceparent" in nats_headers
    assert "traceparent" in temporal_headers
