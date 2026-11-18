"""Spans for calls to Qdrant.

Every call the vector service makes through ``AsyncQdrantClient`` gets its
span here. The OpenTelemetry Qdrant auto-instrumentation is not used: it does
not wrap the async client's ``query_points`` or ``query_batch_points``, and it
ends the spans of the async calls it does wrap before they finish.
"""

import logging
from contextlib import AbstractContextManager
from typing import Any

from opentelemetry import trace

_logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("echora.qdrant_db")


def qdrant_span(
    operation: str, collection_name: str | None = None
) -> AbstractContextManager[trace.Span]:
    """Start a CLIENT span named ``qdrant.<operation>`` for one Qdrant call.

    Use it around the awaited call so the span lasts as long as the call. A
    call that raises is recorded as an error.

    Args:
        operation: Name of the ``AsyncQdrantClient`` method, e.g. ``"scroll"``.
        collection_name: Collection the call works on, when it has one.

    Returns:
        Context manager that yields the active span.
    """
    attributes = {"db.system": "qdrant", "db.operation.name": operation}
    if collection_name is not None:
        attributes["db.collection.name"] = collection_name
    return _tracer.start_as_current_span(
        f"qdrant.{operation}", kind=trace.SpanKind.CLIENT, attributes=attributes
    )


def emit_telemetry(emit: Any, operation: str) -> None:
    """Emit a telemetry metric safely without breaking search calls.

    Args:
        emit: Zero-argument callable recording the metric.
        operation: Operation label for debug logging on failure.
    """
    # Telemetry must never mask real exceptions or fail search paths.
    try:
        emit()
    except Exception:
        _logger.debug(f"Telemetry emission failed for {operation}", exc_info=True)
