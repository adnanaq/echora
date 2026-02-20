"""Type stubs for hishel._core.models module."""

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from hishel._core._headers import Headers

class EntryMeta:
    """Cache entry metadata."""

    created_at: float
    deleted_at: float | None

    def __init__(self, created_at: float, deleted_at: float | None = None) -> None:
        """
        Initialize an EntryMeta containing creation and optional deletion timestamps.

        Parameters:
                created_at (float): Unix timestamp when the entry was created.
                deleted_at (Optional[float]): Unix timestamp when the entry was deleted, or `None` if not deleted.
        """
        ...

class Request:
    """HTTP request model."""

    method: str
    url: str
    headers: dict[str, str] | Headers
    stream: AsyncIterator[bytes] | None
    metadata: dict[str, Any]

    def __init__(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | Headers,
        stream: AsyncIterator[bytes] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a Request model representing an HTTP request used by the cache/system.

        Parameters:
                method (str): HTTP method (e.g., "GET", "POST").
                url (str): Request URL.
                headers (Union[Dict[str, str], Headers]): Header mapping or Headers instance for the request.
                stream (Optional[AsyncIterator[bytes]]): Optional async iterator of raw bytes providing the request body.
                metadata (Optional[Dict[str, Any]]): Optional ancillary metadata associated with the request.
        """
        ...

class Response:
    """HTTP response model."""

    status_code: int
    headers: dict[str, str] | Headers
    stream: Iterator[bytes] | AsyncIterator[bytes] | None
    metadata: dict[str, Any]

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | Headers,
        stream: Iterator[bytes] | AsyncIterator[bytes] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a Response model representing an HTTP response for the cache system.

        Parameters:
            status_code (int): HTTP status code of the response.
            headers (Union[Dict[str, str], Headers]): Response headers as a plain dict or a Headers instance.
            stream (Optional[Union[Iterator[bytes], AsyncIterator[bytes]]]): Optional synchronous or asynchronous byte stream for the response body.
            metadata (Optional[Dict[str, Any]]): Optional ancillary metadata associated with the response.
        """
        ...

class Entry:
    """Cache entry model."""

    id: uuid.UUID
    request: Request
    response: Response | None
    meta: EntryMeta
    cache_key: bytes

    def __init__(
        self,
        id: uuid.UUID,
        request: Request,
        response: Response | None,
        meta: EntryMeta,
        cache_key: bytes,
    ) -> None:
        """
        Create an Entry representing a cached request/response pair.

        Parameters:
            id (uuid.UUID): Unique identifier for the cache entry.
            request (Request): The stored request.
            response (Optional[Response]): The stored response, or `None` if missing.
            meta (EntryMeta): Metadata for creation and deletion timestamps.
            cache_key (bytes): Byte sequence used as the cache lookup key.
        """
        ...
