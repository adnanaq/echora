"""Helpers for mapping internal Python payloads to protobuf value types."""

from __future__ import annotations

from typing import Any

from google.protobuf.struct_pb2 import Struct


def struct_from_dict(data: dict[str, Any]) -> Struct:
    """Convert a Python dictionary into a protobuf ``Struct``.

    Args:
        data: JSON-like dictionary.

    Returns:
        Protobuf ``Struct`` value. Non-JSON-compatible values are stringified
        as a best-effort fallback.
    """
    struct_value = Struct()
    try:
        struct_value.update(data)
    except Exception:
        # Best-effort fallback for non-JSON values.
        struct_value.update(
            {
                key: (
                    value
                    if isinstance(value, (str, int, float, bool, list, dict))
                    else str(value)
                )
                for key, value in data.items()
            }
        )
    return struct_value

