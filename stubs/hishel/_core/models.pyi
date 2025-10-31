"""Type stubs for hishel._core.models module."""

import uuid
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union

from hishel._core._headers import Headers

class EntryMeta:
    """Cache entry metadata."""
    created_at: float
    deleted_at: Optional[float]

    def __init__(
        self, created_at: float, deleted_at: Optional[float] = None
    ) -> None: ...

class Request:
    """HTTP request model."""
    method: str
    url: str
    headers: Union[Dict[str, str], Headers]
    stream: Optional[Iterator[bytes]]
    metadata: Dict[str, Any]

    def __init__(
        self,
        method: str,
        url: str,
        headers: Union[Dict[str, str], Headers],
        stream: Optional[Iterator[bytes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

class Response:
    """HTTP response model."""
    status_code: int
    headers: Union[Dict[str, str], Headers]
    stream: Optional[Union[Iterator[bytes], AsyncIterator[bytes]]]
    metadata: Dict[str, Any]

    def __init__(
        self,
        status_code: int,
        headers: Union[Dict[str, str], Headers],
        stream: Optional[Union[Iterator[bytes], AsyncIterator[bytes]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

class Entry:
    """Cache entry model."""
    id: uuid.UUID
    request: Request
    response: Optional[Response]
    meta: EntryMeta
    cache_key: bytes

    def __init__(
        self,
        id: uuid.UUID,
        request: Request,
        response: Optional[Response],
        meta: EntryMeta,
        cache_key: bytes,
    ) -> None: ...
