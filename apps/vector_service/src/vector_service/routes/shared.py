"""Provide shared helpers for vector_service routes.

This module currently defines a shared `ErrorDetails` constructor used by
multiple vector_service route handlers.
"""

from __future__ import annotations

import json
from typing import Any

from vector_proto.v1 import vector_common_pb2


def error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> vector_common_pb2.ErrorDetails:
    """Build a normalized protobuf error payload.

    Args:
        code: Stable machine-readable error code.
        message: Human-readable error description.
        retryable: Whether callers may retry safely.
        details: Optional structured metadata encoded as JSON.

    Returns:
        Populated protobuf error message.
    """
    return vector_common_pb2.ErrorDetails(
        code=code,
        message=message,
        retryable=retryable,
        details_json=json.dumps(details or {}),
    )
