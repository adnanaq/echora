"""Auto-instrumentation helpers for supported client/server runtimes."""

import logging

logger = logging.getLogger(__name__)
_GRPC_SERVER_INSTRUMENTED = False
_GRPC_CLIENT_INSTRUMENTED = False
_AIOHTTP_CLIENT_INSTRUMENTED = False
_QDRANT_CLIENT_INSTRUMENTED = False
_REDIS_INSTRUMENTED = False


def instrument_grpc_server() -> None:
    """Enable OpenTelemetry auto-instrumentation for the gRPC server.

    Installs ``GrpcInstrumentorServer`` which creates SERVER-kind spans for
    every inbound RPC automatically. Idempotent — safe to call more than once.

    Note:
        Must be called **before** ``grpc.aio.server()`` is created. If called
        after server construction, existing handlers will not be instrumented.
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
    """Enable OpenTelemetry auto-instrumentation for gRPC client stubs.

    Installs ``GrpcInstrumentorClient`` which creates CLIENT-kind spans and
    injects ``traceparent`` headers into outbound gRPC calls. Idempotent.
    """
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
    """Enable OpenTelemetry auto-instrumentation for aiohttp client sessions.

    Installs ``AioHttpClientInstrumentor`` which creates CLIENT-kind spans for
    every outbound HTTP request and propagates trace context via headers.
    Idempotent.
    """
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
    """Enable OpenTelemetry auto-instrumentation for the Qdrant client.

    Installs ``QdrantInstrumentor`` which creates spans for Qdrant operations
    (search, upsert, delete, etc.) and records query metadata as span
    attributes. Idempotent.
    """
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

def instrument_redis() -> None:
    """Enable OpenTelemetry auto-instrumentation for Redis clients.

    Installs ``RedisInstrumentor`` which creates spans for all Redis commands
    (GET, SET, etc.) executed by both sync and async clients. This provides
    visibility into cache hit/miss overhead in the enrichment pipeline and
    embedding layers. Idempotent.
    """
    global _REDIS_INSTRUMENTED
    if _REDIS_INSTRUMENTED:
        return

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
    except ImportError:
        logger.warning("Redis instrumentation is unavailable")
        return

    RedisInstrumentor().instrument()
    _REDIS_INSTRUMENTED = True
