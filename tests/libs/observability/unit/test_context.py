from __future__ import annotations

from typing import Any, cast

import pytest
from observability.context import (
    extract_trace_context,
    inject_context_into_nats_headers,
    inject_context_into_temporal_headers,
    inject_trace_context,
)
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def tracer(span_exporter: InMemorySpanExporter) -> trace.Tracer:
    return trace.get_tracer(__name__)


def test_extract_trace_context_injected_headers_returns_parent_context(
    tracer: trace.Tracer,
) -> None:

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


def test_extract_trace_context_missing_headers_returns_no_parent(
    tracer: trace.Tracer,
) -> None:
    extracted_context = extract_trace_context({})

    with tracer.start_as_current_span("root", context=extracted_context) as span:
        assert cast(Any, span).parent is None


def test_extract_trace_context_invalid_traceparent_returns_no_parent(
    tracer: trace.Tracer,
) -> None:
    extracted_context = extract_trace_context({"traceparent": "invalid"})

    with tracer.start_as_current_span("root", context=extracted_context) as span:
        assert cast(Any, span).parent is None


def test_inject_context_into_nats_and_temporal_headers_active_span_adds_traceparent(
    tracer: trace.Tracer,
) -> None:

    with tracer.start_as_current_span("producer"):
        nats_headers = inject_context_into_nats_headers({})
        temporal_headers = inject_context_into_temporal_headers({})

    assert "traceparent" in nats_headers
    assert "traceparent" in temporal_headers
