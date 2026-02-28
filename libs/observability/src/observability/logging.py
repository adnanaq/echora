"""Structured logging configuration with structlog and OpenTelemetry integration."""

import logging
import queue
import random
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
    (
        re.compile(r"(?i)(api[_-]?key|authorization|token)\s*[:=]\s*[^\s,;]+"),
        r"\1: [REDACTED]",
    ),
    (re.compile(r"(?i)password\s*[:=]\s*[^\s,;]+"), "password: [REDACTED]"),
    (re.compile(r"(?i)email\s*[:=]\s*[\w\.-]+@[\w\.-]+\.\w+"), "email: [REDACTED]"),
]

_LOG_LISTENER: QueueListener | None = None

# Log levels that are always passed through regardless of sample rate.
_UNSAMPLED_LEVELS = frozenset({"warning", "error", "critical"})


def _make_log_sampler(sample_rate: float) -> Processor:
    """Return a structlog processor that probabilistically drops INFO/DEBUG logs.

    WARN/ERROR/CRITICAL events are always emitted regardless of sample_rate.
    At millions of requests/day, 10 % INFO sampling (sample_rate=0.1) reduces
    log volume by ~10× while preserving all actionable signals.

    Args:
        sample_rate: Fraction of INFO/DEBUG events to keep (0.0–1.0).
            1.0 means keep all (default for local development).
    """
    if sample_rate >= 1.0:
        # Fast path: return an identity processor — no overhead, no branching.
        def _passthrough(
            logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
        ) -> MutableMapping[str, Any]:
            return event_dict

        return _passthrough

    def _sampler(
        logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        # method_name is the log method called ("info", "warning", "error", etc.)
        # — use it directly rather than reading from event_dict to avoid
        # depending on add_log_level running first.
        if method_name not in _UNSAMPLED_LEVELS and random.random() >= sample_rate:  # noqa: S311
            raise structlog.DropEvent()
        return event_dict

    return _sampler


def redact_pii(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Redact PII patterns from all string values in a log event.

    Applies regex substitutions for common PII patterns (API keys, passwords,
    email addresses) against every string-valued field in the event dict.

    Args:
        logger: The bound logger instance (unused, required by structlog API).
        method_name: Log method name (e.g. ``"info"``). Unused here.
        event_dict: The mutable structlog event dict to redact in-place.

    Returns:
        The same ``event_dict`` with PII values replaced by ``[REDACTED]``.
    """
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern, repl in _PII_PATTERNS:
                event_dict[key] = pattern.sub(repl, event_dict[key])
    return event_dict


def inject_otel_context(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Inject the active OTel trace and span IDs into the log event dict.

    Adds ``trace_id`` and ``span_id`` fields only when there is a valid,
    recording span in the current context. This enables log-to-trace
    correlation in Grafana (Loki → Tempo) without duplicating fields that
    OTel structured metadata already carries.

    Args:
        logger: The bound logger instance (unused, required by structlog API).
        method_name: Log method name (e.g. ``"info"``). Unused here.
        event_dict: The mutable structlog event dict to enrich.

    Returns:
        The same ``event_dict``, with ``trace_id`` and ``span_id`` added when
        a valid active span exists.
    """
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
    log_sample_rate: float = 1.0,
) -> None:
    """Configures structlog for JSON production-grade logging.

    Args:
        level: The logging level to use.
        service_name: Name of the emitting service.
        environment: Runtime environment label.
        endpoint: OTLP collector endpoint.
        log_sample_rate: Fraction of INFO/DEBUG log events to keep (0.0–1.0).
            WARN/ERROR/CRITICAL are never sampled out. Defaults to 1.0 (keep
            all). Set to 0.1 in high-throughput production environments to
            reduce log volume by ~10× without losing actionable signals.
    """
    global _LOG_LISTENER

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        # Sampling runs after level is available but before any I/O or
        # formatting — dropped events incur zero serialisation cost.
        _make_log_sampler(log_sample_rate),
        inject_otel_context,
        redact_pii,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    # service, env, timestamp, and logger_name are omitted from the body:
    # they are captured by OTel as Resource attributes (service.name,
    # deployment.environment), ObservedTimestamp, and InstrumentationScope
    # respectively. Duplicating them in the JSON body creates redundant fields
    # in Loki structured metadata.
    structlog.contextvars.clear_contextvars()

    # 1. Setup OpenTelemetry Log Bridge
    resource = get_aggregated_resources(
        detectors=[],
        initial_resource=Resource.create(
            {"service.name": service_name, "deployment.environment": environment}
        ),
    )
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)

    exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    # This handler bridges Python logging to OpenTelemetry
    otel_handler = LoggingHandler(
        level=getattr(logging, level.upper()), logger_provider=logger_provider
    )

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

    # otel_handler is added directly to the root logger (not via the queue) so
    # that it fires synchronously in the calling thread/task where the OTel span
    # context is still active.  BatchLogRecordProcessor queues internally and is
    # non-blocking, so this does not introduce latency in the hot path.
    # If otel_handler ran inside QueueListener (a background thread), the span
    # context would be lost and trace_id/span_id would never be captured.
    root_logger.addHandler(otel_handler)

    # QueueHandler + QueueListener handles non-blocking stdout output only.
    root_logger.addHandler(QueueHandler(log_queue))

    # Start the listener in a background thread
    if _LOG_LISTENER is not None:
        _LOG_LISTENER.stop()

    _LOG_LISTENER = QueueListener(
        log_queue, stdout_handler, respect_handler_level=True
    )
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
