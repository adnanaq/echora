"""Trace context helpers for transport boundaries."""

from collections.abc import Mapping, MutableMapping

from opentelemetry import context as otel_context
from opentelemetry import propagate


def inject_trace_context(
    carrier: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject current trace context into a generic string carrier."""
    target = {} if carrier is None else dict(carrier)
    propagate.inject(target)
    return target


def extract_trace_context(
    carrier: Mapping[str, str] | None,
) -> otel_context.Context:
    """Extract trace context from a generic string carrier."""
    return propagate.extract({} if carrier is None else dict(carrier))


def inject_context_into_nats_headers(
    headers: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject trace context into NATS-compatible headers."""
    return inject_trace_context(headers)


def extract_context_from_nats_headers(
    headers: Mapping[str, str] | None,
) -> otel_context.Context:
    """Extract trace context from NATS-compatible headers."""
    return extract_trace_context(headers)


def inject_context_into_temporal_headers(
    headers: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject trace context into Temporal-compatible headers."""
    return inject_trace_context(headers)


def extract_context_from_temporal_headers(
    headers: Mapping[str, str] | None,
) -> otel_context.Context:
    """Extract trace context from Temporal-compatible headers."""
    return extract_trace_context(headers)
