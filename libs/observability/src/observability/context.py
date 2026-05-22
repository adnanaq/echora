"""Trace context helpers for transport boundaries."""

from collections.abc import Mapping, MutableMapping

from opentelemetry import context as otel_context
from opentelemetry import propagate


def inject_trace_context(
    carrier: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject the active trace context into a string carrier dict.

    Uses the globally configured W3C TraceContext propagator so the
    ``traceparent``/``tracestate`` headers are written automatically.

    Args:
        carrier: Existing mapping to inject into. A fresh ``dict`` is used
            when ``None`` is passed.

    Returns:
        A new ``dict`` containing the original carrier contents plus the
        injected trace headers.
    """
    target = {} if carrier is None else dict(carrier)
    propagate.inject(target)
    return target


def extract_trace_context(
    carrier: Mapping[str, str] | None,
) -> otel_context.Context:
    """Extract a trace context from a string carrier dict.

    Args:
        carrier: Mapping that may contain ``traceparent``/``tracestate``
            headers. Treated as empty when ``None``.

    Returns:
        An OTel ``Context`` object. If no valid trace headers are present,
        a new empty context is returned (no exception is raised).
    """
    return propagate.extract({} if carrier is None else dict(carrier))


def inject_context_into_nats_headers(
    headers: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject the active trace context into NATS message headers.

    Args:
        headers: Existing NATS headers mapping to inject into. A fresh
            ``dict`` is used when ``None`` is passed.

    Returns:
        A new ``dict`` containing the original headers plus trace headers.
    """
    return inject_trace_context(headers)


def extract_context_from_nats_headers(
    headers: Mapping[str, str] | None,
) -> otel_context.Context:
    """Extract a trace context from NATS message headers.

    Args:
        headers: NATS headers that may contain ``traceparent``/``tracestate``.
            Treated as empty when ``None``.

    Returns:
        An OTel ``Context`` extracted from the headers.
    """
    return extract_trace_context(headers)


def inject_context_into_temporal_headers(
    headers: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject the active trace context into Temporal workflow headers.

    Args:
        headers: Existing Temporal header mapping to inject into. A fresh
            ``dict`` is used when ``None`` is passed.

    Returns:
        A new ``dict`` containing the original headers plus trace headers.
    """
    return inject_trace_context(headers)


def extract_context_from_temporal_headers(
    headers: Mapping[str, str] | None,
) -> otel_context.Context:
    """Extract a trace context from Temporal workflow headers.

    Args:
        headers: Temporal headers that may contain ``traceparent``/``tracestate``.
            Treated as empty when ``None``.

    Returns:
        An OTel ``Context`` extracted from the headers.
    """
    return extract_trace_context(headers)
