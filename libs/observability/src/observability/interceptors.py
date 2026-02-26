"""Async gRPC interceptors for telemetry and domain monitoring."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

import grpc
from opentelemetry import trace
from .registry import registry, _TRACER


class AioServerInterceptor(grpc.aio.ServerInterceptor):
    """Asynchronous gRPC server interceptor for telemetry.

    This interceptor handles:
    1. RPC request/error counting.
    2. RPC duration recording (Histogram).
    3. Automatic span creation (in addition to auto-instrumentation).
    4. Pattern-aware business logic failure detection.
    """

    async def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
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
        async def new_behavior(request: Any, context: grpc.aio.ServicerContext) -> Any:
            attrs = {"rpc_method": method_name}
            registry.RPC_REQUESTS.add(1, attrs)
            start_time = time.perf_counter()

            # We use start_as_current_span to ensure this span is linked to the 
            # incoming trace context (handled by GrpcInstrumentorServer automatically)
            with _TRACER.start_as_current_span(
                f"rpc.server.{method_name}",
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
                    
                    return response
                except Exception as exc:
                    registry.RPC_ERRORS.add(1, {**attrs, "error_code": "EXCEPTION"})
                    span.record_exception(exc)
                    span.set_attribute("rpc.error", True)
                    span.set_attribute("rpc.error_code", "EXCEPTION")
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    registry.RPC_DURATION.record(duration, attrs)

        return new_behavior
