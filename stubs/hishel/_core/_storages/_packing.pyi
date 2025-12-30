"""Type stubs for hishel._core._storages._packing module."""

from typing import Any, Literal

def pack(obj: Any, kind: Literal["pair"] = "pair") -> bytes:
    """
    Serialize a cache entry to bytes using MessagePack.

    Parameters:
        obj (Any): Object to serialize (typically an Entry).
        kind (Literal["pair"], optional): Serialization variant to use; defaults to "pair".

    Returns:
        bytes: Serialized bytes.
    """
    ...

def unpack(data: bytes, kind: Literal["pair"] = "pair") -> Any:
    """
    Deserialize `data` into a cache entry using msgpack.

    Parameters:
        data (bytes): Serialized bytes produced by the corresponding pack function.
        kind (Literal["pair"], optional): Deserialization format selector; only "pair" is supported by this stub. Defaults to "pair".

    Returns:
        Any: The deserialized object (typically an Entry).
    """
    ...
