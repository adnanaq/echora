"""Async gRPC interceptors for telemetry and domain monitoring."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

import grpc
import structlog
from opentelemetry import propagate, trace

from .registry import _TRACER, registry

_log = structlog.get_logger(__name__)


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

        if handler and handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_unary(handler.unary_unary, method_name),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        # TODO: Add wrappers for streaming handlers if needed
        return handler

    def _wrap_unary(self, behavior: Callable, method_name: str) -> Callable:
        """Wrap a unary RPC handler with telemetry instrumentation.

        Args:
            behavior: The original unary RPC handler callable.
            method_name: Short RPC method name (e.g. ``"Search"``), used as
                the span name suffix and metric label.

        Returns:
            A new async callable with the same signature as ``behavior`` that
            records metrics, creates a SERVER span, and emits an error log on
            RPC failure before delegating to the original handler.
        """
        async def new_behavior(request: Any, context: grpc.aio.ServicerContext) -> Any:
            attrs = {"rpc_method": method_name}
            registry.RPC_REQUESTS.add(1, attrs)
            registry.INFLIGHT_RPCS.add(1, attrs)
            start_time = time.perf_counter()

            # Extract W3C traceparent/tracestate from incoming gRPC metadata.
            # This links child spans to whatever trace the upstream caller started,
            # enabling end-to-end distributed traces across service boundaries.
            # Only string-typed metadata keys are used; binary keys (e.g.
            # "grpc-trace-bin") are intentionally excluded.
            metadata: dict[str, str] = {
                k: v
                for k, v in (context.invocation_metadata() or ())
                if isinstance(k, str) and isinstance(v, str)
            }
            parent_ctx = propagate.extract(metadata)

            # Bind the method name to structlog's per-task ContextVar so that
            # every log line emitted during this RPC includes rpc_method.
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

                    # --- ROBUST FAILURE DETECTION (CONTRACT-AWARE) ---
                    is_failure = False
                    error_code = "UNKNOWN"

                    # 1. Check gRPC status code (transport failure)
                    code = context.code()
                    if code is not None and code != grpc.StatusCode.OK:
                        is_failure = True
                        error_code = code.name

                    # 2. Check for explicit success boolean (Enrichment Service)
                    elif hasattr(response, "success") and not response.success:
                        is_failure = True

                    # 3. Check for explicit health boolean (Health RPCs)
                    elif hasattr(response, "healthy") and not response.healthy:
                        is_failure = True

                    # 4. Check for presence of 'error' message (Vector Service / Universal Fallback)
                    elif hasattr(response, "HasField") and hasattr(response, "error"):
                        if response.HasField("error"):
                            is_failure = True
                            # Extract specific error code from the ErrorDetails object
                            error_code = getattr(response.error, "code", "UNKNOWN")

                    if is_failure:
                        registry.RPC_ERRORS.add(1, {**attrs, "error_code": error_code})
                        span.set_attribute("rpc.error", True)
                        span.set_attribute("rpc.error_code", error_code)
                        _log.error("rpc.failure", error_code=error_code)
                except Exception as exc:
                    registry.RPC_ERRORS.add(1, {**attrs, "error_code": "EXCEPTION"})
                    span.record_exception(exc)
                    span.set_attribute("rpc.error", True)
                    span.set_attribute("rpc.error_code", "EXCEPTION")
                    raise
                else:
                    return response
                finally:
                    duration = time.perf_counter() - start_time
                    registry.INFLIGHT_RPCS.add(-1, attrs)
                    registry.RPC_DURATION.record(duration, attrs)
                    structlog.contextvars.unbind_contextvars("rpc_method")

        return new_behavior
