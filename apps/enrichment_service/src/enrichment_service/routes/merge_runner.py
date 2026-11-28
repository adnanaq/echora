"""Shared async execution and error-mapping helper for merge gRPC routes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TypeVar

import grpc

logger = logging.getLogger(__name__)

_RequestT = TypeVar("_RequestT")
_ResponseT = TypeVar("_ResponseT")


async def execute_merge_rpc(
    merge_function: Callable[[_RequestT], _ResponseT],
    request: _RequestT,
    context: grpc.aio.ServicerContext,
    operation_name: str,
) -> _ResponseT:
    """Run a CPU-bound merge off the event loop and map errors to gRPC statuses.

    Malformed stored payloads and validation errors (``ValueError``, ``TypeError``)
    are logged at ``WARNING`` and aborted with ``INVALID_ARGUMENT`` so Temporal does
    not retry unrecoverable input errors. Unexpected failures are logged with full
    tracebacks and aborted with ``INTERNAL``.

    Args:
        merge_function: Pure synchronous merge callable taking ``request``.
        request: Incoming protobuf merge request message.
        context: Active gRPC servicer context for status reporting.
        operation_name: RPC name used in logs and error messages.

    Returns:
        Protobuf response produced by ``merge_function``.
    """
    try:
        return await asyncio.to_thread(merge_function, request)
    except (ValueError, TypeError) as exc:
        logger.warning(f"{operation_name} rejected invalid input: {exc}")
        return await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    except Exception:
        logger.exception(f"{operation_name} failed unexpectedly")
        return await context.abort(grpc.StatusCode.INTERNAL, f"{operation_name} failed")
