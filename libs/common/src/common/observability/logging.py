"""Structured logging configuration with structlog and OpenTelemetry integration."""

import logging
from typing import Any

import structlog
from opentelemetry import trace


def inject_otel_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Injects OpenTelemetry trace and span IDs into the log event."""
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = hex(ctx.trace_id)[2:].zfill(32)
            event_dict["span_id"] = hex(ctx.span_id)[2:].zfill(16)
    return event_dict


def setup_logging(
    *,
    level: str = "INFO",
    service_name: str = "echora-service",
    environment: str = "development",
) -> None:
    """Configures structlog for JSON production-grade logging.

    Args:
        level: The logging level to use.
        service_name: Name of the emitting service.
        environment: Runtime environment label.
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        inject_otel_context,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    # Bind common context values once so all future events include service/env.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name, env=environment)

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        force=True,
    )

    # Ensure stdlib loggers produce structured output through structlog processors.
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
