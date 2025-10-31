"""Type stubs for hishel._core._storages._packing module."""

from typing import Any, Literal

def pack(obj: Any, kind: Literal["pair"] = "pair") -> bytes:
    """
    Serialize a cache entry to bytes using msgpack.

    Args:
        obj: Object to serialize (typically Entry)
        kind: Serialization type (default: "pair")

    Returns:
        Serialized bytes
    """
    ...

def unpack(data: bytes, kind: Literal["pair"] = "pair") -> Any:
    """
    Deserialize bytes to cache entry using msgpack.

    Args:
        data: Serialized bytes
        kind: Serialization type (default: "pair")

    Returns:
        Deserialized object (typically Entry)
    """
    ...
