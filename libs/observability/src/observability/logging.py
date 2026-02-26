"""Structured logging configuration with structlog and OpenTelemetry integration."""

import logging
import queue
import re
from collections.abc import MutableMapping
from logging.handlers import QueueHandler, QueueListener
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource, get_aggregated_resources
from structlog.types import Processor

# Common PII patterns for redaction
_PII_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|authorization|token)\s*[:=]\s*[^\s,;]+"), r"\1: [REDACTED]"),
    (re.compile(r"(?i)password\s*[:=]\s*[^\s,;]+"), "password: [REDACTED]"),
    (re.compile(r"(?i)email\s*[:=]\s*[\w\.-]+@[\w\.-]+\.\w+"), "email: [REDACTED]"),
]

_LOG_LISTENER: QueueListener | None = None


def redact_pii(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Redacts PII from log events."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern, repl in _PII_PATTERNS:
                event_dict[key] = pattern.sub(repl, event_dict[key])
    return event_dict


def inject_otel_context(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
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
    endpoint: str = "http://localhost:4317",
) -> None:
    """Configures structlog for JSON production-grade logging.

    Args:
        level: The logging level to use.
        service_name: Name of the emitting service.
        environment: Runtime environment label.
        endpoint: OTLP collector endpoint.
    """
    global _LOG_LISTENER

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        inject_otel_context,
        redact_pii,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    # Bind common context values once so all future events include service/env.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name, env=environment)

    # 1. Setup OpenTelemetry Log Bridge
    resource = get_aggregated_resources(
        detectors=[],
        initial_resource=Resource.create({"service.name": service_name, "deployment.environment": environment}),
    )
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    
    exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    
    # This handler bridges Python logging to OpenTelemetry
    otel_handler = LoggingHandler(level=getattr(logging, level.upper()), logger_provider=logger_provider)

    # 2. Setup non-blocking logging using QueueHandler
    log_queue: queue.Queue = queue.Queue(-1)
    
    # Base handler that actually writes to stdout
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    
    # Add QueueHandler to root logger
    root_logger.addHandler(QueueHandler(log_queue))
    
    # Start the listener in a background thread
    if _LOG_LISTENER is not None:
        _LOG_LISTENER.stop()
    
    # The listener flushes to BOTH stdout and the OTel bridge
    _LOG_LISTENER = QueueListener(log_queue, stdout_handler, otel_handler, respect_handler_level=True)
    _LOG_LISTENER.start()

    # Ensure stdlib loggers produce structured output through structlog processors.
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def stop_logging() -> None:
    """Gracefully stop the background logging listener."""
    global _LOG_LISTENER
    if _LOG_LISTENER:
        _LOG_LISTENER.stop()
        _LOG_LISTENER = None
