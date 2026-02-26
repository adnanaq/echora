"""Auto-instrumentation helpers for supported client/server runtimes."""

import logging

logger = logging.getLogger(__name__)
_GRPC_SERVER_INSTRUMENTED = False
_GRPC_CLIENT_INSTRUMENTED = False
_AIOHTTP_CLIENT_INSTRUMENTED = False
_QDRANT_CLIENT_INSTRUMENTED = False


def instrument_grpc_server() -> None:
    """Enable gRPC server instrumentation when available.
    IMPORTANT: This should be called BEFORE creating the grpc.aio.server() instance.
    """
    global _GRPC_SERVER_INSTRUMENTED
    if _GRPC_SERVER_INSTRUMENTED:
        return

    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
    except ImportError:
        logger.warning("gRPC server instrumentation is unavailable")
        return

    GrpcInstrumentorServer().instrument()
    _GRPC_SERVER_INSTRUMENTED = True


def instrument_grpc_client() -> None:
    """Enable gRPC client instrumentation when available."""
    global _GRPC_CLIENT_INSTRUMENTED
    if _GRPC_CLIENT_INSTRUMENTED:
        return

    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
    except ImportError:
        logger.warning("gRPC client instrumentation is unavailable")
        return

    GrpcInstrumentorClient().instrument()
    _GRPC_CLIENT_INSTRUMENTED = True


def instrument_aiohttp_client() -> None:
    """Enable aiohttp client instrumentation when available."""
    global _AIOHTTP_CLIENT_INSTRUMENTED
    if _AIOHTTP_CLIENT_INSTRUMENTED:
        return

    try:
        from opentelemetry.instrumentation.aiohttp_client import (
            AioHttpClientInstrumentor,
        )
    except ImportError:
        logger.warning("aiohttp client instrumentation is unavailable")
        return

    AioHttpClientInstrumentor().instrument()
    _AIOHTTP_CLIENT_INSTRUMENTED = True


def instrument_qdrant_client() -> None:
    """Enable Qdrant client instrumentation when available."""
    global _QDRANT_CLIENT_INSTRUMENTED
    if _QDRANT_CLIENT_INSTRUMENTED:
        return

    try:
        from opentelemetry.instrumentation.qdrant import QdrantInstrumentor
    except ImportError:
        logger.warning("Qdrant client instrumentation is unavailable")
        return

    QdrantInstrumentor().instrument()
    _QDRANT_CLIENT_INSTRUMENTED = True
