"""Vector service RPC telemetry helpers.

This module provides lightweight tracing and metrics wrappers so dashboard
signals are available even when runtime auto-instrumentation is partial.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import metrics, trace

_TRACER = trace.get_tracer("vector_service.routes")
_METER = metrics.get_meter("vector_service.routes")

_RPC_REQUESTS = _METER.create_counter(
    "echora_vector_service_rpc_requests_total",
    description="Total RPC requests handled by vector service",
)
_RPC_ERRORS = _METER.create_counter(
    "echora_vector_service_rpc_errors_total",
    description="Total RPC requests that ended in error",
)
_RPC_DURATION = _METER.create_histogram(
    "echora_vector_service_rpc_duration_seconds",
    unit="s",
    description="RPC handler duration in seconds",
)


def mark_rpc_error(rpc_method: str, *, code: str | None = None) -> None:
    """Record an RPC-level error outcome."""
    attrs = {"rpc_method": rpc_method}
    if code:
        attrs["error_code"] = code
    _RPC_ERRORS.add(1, attrs)


@contextmanager
def rpc_span(rpc_method: str) -> Iterator[object]:
    """Create a server span and emit request/latency metrics."""
    attrs = {"rpc_method": rpc_method}
    _RPC_REQUESTS.add(1, attrs)
    start = time.perf_counter()
    with _TRACER.start_as_current_span(f"rpc.{rpc_method}") as span:
        span.set_attribute("rpc.system", "grpc")
        span.set_attribute("rpc.method", rpc_method)
        try:
            yield span
        except Exception as exc:
            _RPC_ERRORS.add(1, attrs)
            span.record_exception(exc)
            span.set_attribute("rpc.error", True)
            raise
        finally:
            _RPC_DURATION.record(time.perf_counter() - start, attrs)
