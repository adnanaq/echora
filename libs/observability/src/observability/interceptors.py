"""Async gRPC interceptors for telemetry and domain monitoring."""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import grpc
import structlog
from opentelemetry import propagate, trace

from .registry import _TRACER, registry

_log = structlog.get_logger(__name__)

# Predefined list of error codes that are safe to use as Prometheus labels.
# Any code not in this list will be mapped to "OTHER_ERROR" for metrics only,
# while the full detailed code is still preserved in traces (Tempo) and logs (Loki).
_SAFE_ERROR_CODES = frozenset(
    [
        # Standard gRPC Status Codes
        "OK",
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED",
        # Generic Failures
        "FAILURE",
        "UNHEALTHY",
        "EXCEPTION",
        # Service domain codes
        "INVALID_FILE_PATH",
        "INVALID_AGENT_DIR",
        "RUN_PIPELINE_FAILED",
        "TEXT_EMBEDDING_FAILED",
        "IMAGE_EMBEDDING_FAILED",
        "QDRANT_SEARCH_FAILED",
        "QDRANT_UPSERT_FAILED",
        "HEALTH_FAILED",
        "GET_STATS_FAILED",
        "MISSING_QUERY_INPUT",
        "INVALID_FILTERS",
        "SEARCH_FAILED",
    ]
)


class AioServerInterceptor(grpc.aio.ServerInterceptor):
    """Asynchronous gRPC server interceptor for telemetry.

    Per-RPC responsibilities:
    1. Extract incoming W3C traceparent/tracestate from gRPC metadata so that
       upstream callers (API gateways, other services) are linked into the same
       distributed trace.
    2. Count RPC requests, errors, and in-flight calls.
    3. Record RPC duration as a histogram observation.
    4. Create a SERVER-kind span for the method, parented to the upstream trace.
    5. Bind rpc_method to structlog contextvars so every log line emitted during
       the RPC automatically includes the method name.
    6. Detect failures via a contract-aware multi-level check.
    """

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Intercepts gRPC service calls."""
        method = handler_call_details.method
        # method usually looks like '/package.Service/Method'
        method_name = method.split("/")[-1] if "/" in method else method

        handler = await continuation(handler_call_details)
        if not handler:
            return handler

        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_unary(handler.unary_unary, method_name),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream(handler.unary_stream, method_name),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary(handler.stream_unary, method_name),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                self._wrap_stream_stream(handler.stream_stream, method_name),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        return handler

    def _sanitize_error_code(self, code: str) -> str:
        """Protect against cardinality explosion by limiting metric label values."""
        if code in _SAFE_ERROR_CODES:
            return code
        return "OTHER_ERROR"

    def _record_failure(self, span: trace.Span, attrs: dict, error_code: str) -> None:
        """Record failure metrics, span attributes, and log the event."""
        safe_code = self._sanitize_error_code(error_code)
        registry.RPC_ERRORS.add(1, {**attrs, "error_code": safe_code})
        span.set_attribute("rpc.error", True)
        span.set_attribute("rpc.error_code", error_code)
        _log.error("rpc.failure", error_code=error_code)

    def _detect_error_code(self, context: grpc.aio.ServicerContext, response: Any = None) -> str | None:
        """Detect if RPC failed and return the machine-readable error code, else None."""
        # 1. Check gRPC status code (transport failure)
        code = context.code()
        if code is not None and code != grpc.StatusCode.OK:
            return code.name

        if response is not None:
            # 2. Check for explicit success boolean (Enrichment Service)
            if hasattr(response, "success") and not response.success:
                return getattr(response.error, "code", "FAILURE") if hasattr(response, "error") else "FAILURE"

            # 3. Check for explicit health boolean (Health RPCs)
            if hasattr(response, "healthy") and not response.healthy:
                return "UNHEALTHY"

            # 4. Check for presence of 'error' message (Vector Service / Universal Fallback)
            if hasattr(response, "HasField") and hasattr(response, "error"):
                if response.HasField("error"):
                    return getattr(response.error, "code", "UNKNOWN")
        return None

    def _wrap_unary(self, behavior: Callable, method_name: str) -> Callable:
        """Wrap a unary-unary RPC handler with telemetry instrumentation."""
        async def new_behavior(request: Any, context: grpc.aio.ServicerContext) -> Any:
            attrs = {"rpc_method": method_name}
            registry.RPC_REQUESTS.add(1, attrs)
            registry.INFLIGHT_RPCS.add(1, attrs)
            start_time = time.perf_counter()

            metadata: dict[str, str] = {
                k: v
                for k, v in (context.invocation_metadata() or ())
                if isinstance(k, str) and isinstance(v, str)
            }
            parent_ctx = propagate.extract(metadata)
            structlog.contextvars.bind_contextvars(rpc_method=method_name)

            with _TRACER.start_as_current_span(
                f"rpc.server.{method_name}",
                context=parent_ctx,
                kind=trace.SpanKind.SERVER,
            ) as span:
                span.set_attribute("rpc.system", "grpc")
                span.set_attribute("rpc.method", method_name)

                try:
                    response = await behavior(request, context)
                    error_code = self._detect_error_code(context, response)
                    if error_code:
                        self._record_failure(span, attrs, error_code)
                    return response
                except Exception as exc:
                    self._record_failure(span, attrs, "EXCEPTION")
                    span.record_exception(exc)
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    registry.INFLIGHT_RPCS.add(-1, attrs)
                    registry.RPC_DURATION.record(duration, attrs)
                    structlog.contextvars.unbind_contextvars("rpc_method")

        return new_behavior

    def _wrap_unary_stream(self, behavior: Callable, method_name: str) -> Callable:
        """Wrap a unary-stream RPC handler with telemetry instrumentation."""
        async def new_behavior(request: Any, context: grpc.aio.ServicerContext) -> AsyncIterator:
            attrs = {"rpc_method": method_name}
            registry.RPC_REQUESTS.add(1, attrs)
            registry.INFLIGHT_RPCS.add(1, attrs)
            start_time = time.perf_counter()

            metadata: dict[str, str] = {
                k: v
                for k, v in (context.invocation_metadata() or ())
                if isinstance(k, str) and isinstance(v, str)
            }
            parent_ctx = propagate.extract(metadata)
            structlog.contextvars.bind_contextvars(rpc_method=method_name)

            with _TRACER.start_as_current_span(
                f"rpc.server.{method_name}",
                context=parent_ctx,
                kind=trace.SpanKind.SERVER,
            ) as span:
                span.set_attribute("rpc.system", "grpc")
                span.set_attribute("rpc.method", method_name)

                try:
                    async for response in behavior(request, context):
                        yield response

                    # Final check for late errors (e.g. context cancellation)
                    error_code = self._detect_error_code(context)
                    if error_code:
                        self._record_failure(span, attrs, error_code)
                except Exception as exc:
                    self._record_failure(span, attrs, "EXCEPTION")
                    span.record_exception(exc)
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    registry.INFLIGHT_RPCS.add(-1, attrs)
                    registry.RPC_DURATION.record(duration, attrs)
                    structlog.contextvars.unbind_contextvars("rpc_method")

        return new_behavior

    def _wrap_stream_unary(self, behavior: Callable, method_name: str) -> Callable:
        """Wrap a stream-unary RPC handler with telemetry instrumentation."""
        async def new_behavior(request_iterator: AsyncIterator, context: grpc.aio.ServicerContext) -> Any:
            attrs = {"rpc_method": method_name}
            registry.RPC_REQUESTS.add(1, attrs)
            registry.INFLIGHT_RPCS.add(1, attrs)
            start_time = time.perf_counter()

            metadata: dict[str, str] = {
                k: v
                for k, v in (context.invocation_metadata() or ())
                if isinstance(k, str) and isinstance(v, str)
            }
            parent_ctx = propagate.extract(metadata)
            structlog.contextvars.bind_contextvars(rpc_method=method_name)

            with _TRACER.start_as_current_span(
                f"rpc.server.{method_name}",
                context=parent_ctx,
                kind=trace.SpanKind.SERVER,
            ) as span:
                span.set_attribute("rpc.system", "grpc")
                span.set_attribute("rpc.method", method_name)

                try:
                    response = await behavior(request_iterator, context)
                    error_code = self._detect_error_code(context, response)
                    if error_code:
                        self._record_failure(span, attrs, error_code)
                    return response
                except Exception as exc:
                    self._record_failure(span, attrs, "EXCEPTION")
                    span.record_exception(exc)
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    registry.INFLIGHT_RPCS.add(-1, attrs)
                    registry.RPC_DURATION.record(duration, attrs)
                    structlog.contextvars.unbind_contextvars("rpc_method")

        return new_behavior

    def _wrap_stream_stream(self, behavior: Callable, method_name: str) -> Callable:
        """Wrap a stream-stream RPC handler with telemetry instrumentation."""
        async def new_behavior(request_iterator: AsyncIterator, context: grpc.aio.ServicerContext) -> AsyncIterator:
            attrs = {"rpc_method": method_name}
            registry.RPC_REQUESTS.add(1, attrs)
            registry.INFLIGHT_RPCS.add(1, attrs)
            start_time = time.perf_counter()

            metadata: dict[str, str] = {
                k: v
                for k, v in (context.invocation_metadata() or ())
                if isinstance(k, str) and isinstance(v, str)
            }
            parent_ctx = propagate.extract(metadata)
            structlog.contextvars.bind_contextvars(rpc_method=method_name)

            with _TRACER.start_as_current_span(
                f"rpc.server.{method_name}",
                context=parent_ctx,
                kind=trace.SpanKind.SERVER,
            ) as span:
                span.set_attribute("rpc.system", "grpc")
                span.set_attribute("rpc.method", method_name)

                try:
                    async for response in behavior(request_iterator, context):
                        yield response

                    error_code = self._detect_error_code(context)
                    if error_code:
                        self._record_failure(span, attrs, error_code)
                except Exception as exc:
                    self._record_failure(span, attrs, "EXCEPTION")
                    span.record_exception(exc)
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    registry.INFLIGHT_RPCS.add(-1, attrs)
                    registry.RPC_DURATION.record(duration, attrs)
                    structlog.contextvars.unbind_contextvars("rpc_method")

        return new_behavior
