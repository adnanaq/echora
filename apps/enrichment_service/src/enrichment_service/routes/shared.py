"""Shared helpers for enrichment_service gRPC route handlers."""

from __future__ import annotations

import json
from typing import Any

from enrichment_proto.v1 import enrichment_service_pb2


def error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> enrichment_service_pb2.ErrorDetails:
    """Build a normalized protobuf error payload.

    Args:
        code: Stable machine-readable error code.
        message: Human-readable error description.
        retryable: Whether callers may retry safely.
        details: Optional structured metadata encoded as JSON.

    Returns:
        Populated protobuf error message.
    """
    return enrichment_service_pb2.ErrorDetails(
        code=code,
        message=message,
        retryable=retryable,
        details_json=json.dumps(details or {}),
    )
