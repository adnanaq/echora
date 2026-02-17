"""Build common gRPC error payloads for service responses."""

from __future__ import annotations

import json
from typing import Any

from shared_proto.v1 import error_pb2


def build_error_details(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> error_pb2.ErrorDetails:
    """Create a normalized protobuf error payload.

    Args:
        code: Stable machine-readable error code.
        message: Human-readable error description.
        retryable: Whether callers may retry safely.
        details: Optional structured metadata encoded as JSON.

    Returns:
        Populated protobuf error message.
    """
    return error_pb2.ErrorDetails(
        code=code,
        message=message,
        retryable=retryable,
        details_json=json.dumps(details or {}),
    )
