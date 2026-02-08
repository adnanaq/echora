"""Type stubs for hishel HTTP caching library (v1.0)."""

import uuid
from collections.abc import Callable
from typing import Any

# Re-export core data models from internal modules (matches hishel/__init__.py structure)
from hishel._core._headers import Headers as Headers
from hishel._core.models import (
    Entry as Entry,
)
from hishel._core.models import (
    EntryMeta as EntryMeta,
)
from hishel._core.models import (
    Request as Request,
)
from hishel._core.models import (
    Response as Response,
)

# Storage base classes (public API)
#
# NOTE ON PARAMETER NAMING: id vs id_
# - create_entry uses 'id_' (optional parameter, avoids shadowing Python builtin)
# - update_entry and remove_entry use 'id' (required parameter, matches hishel runtime)
# Verified against hishel v1.0 source code.
# DO NOT change without verifying against actual hishel source.
class SyncBaseStorage:
    """Base class for synchronous cache storage implementations."""

    def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create a cache entry.

        Parameters:
            request (Request): The request to cache.
            response (Response): The response to cache.
            key (str): Cache key.
            id_ (Optional[uuid.UUID]): Optional entry ID.

        Returns:
            Entry: The created cache entry.
        """
        ...

    def get_entries(self, key: str) -> list[Entry]:
        """
        Retrieve cache entries for a given key.

        Parameters:
            key (str): Cache key to lookup.

        Returns:
            List[Entry]: List of matching cache entries.
        """
        ...

    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Entry | Callable[[Entry], Entry],
    ) -> Entry | None:
        """
        Update an existing cache entry.

        Parameters:
            id (uuid.UUID): Entry identifier.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): New entry or update function.

        Returns:
            Optional[Entry]: Updated entry if found, None otherwise.
        """
        ...

    def remove_entry(self, id: uuid.UUID) -> None:
        """
        Remove a cache entry.

        Parameters:
            id (uuid.UUID): Entry identifier to remove.
        """
        ...

class AsyncBaseStorage:
    """Base class for asynchronous cache storage implementations."""

    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create a cache entry asynchronously.

        Parameters:
            request (Request): The request to cache.
            response (Response): The response to cache.
            key (str): Cache key.
            id_ (Optional[uuid.UUID]): Optional entry ID.

        Returns:
            Entry: The created cache entry.
        """
        ...

    async def get_entries(self, key: str) -> list[Entry]:
        """
        Retrieve cache entries for a given key asynchronously.

        Parameters:
            key (str): Cache key to lookup.

        Returns:
            List[Entry]: List of matching cache entries.
        """
        ...

    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Entry | Callable[[Entry], Entry],
    ) -> Entry | None:
        """
        Update an existing cache entry asynchronously.

        Parameters:
            id (uuid.UUID): Entry identifier.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): New entry or update function.

        Returns:
            Optional[Entry]: Updated entry if found, None otherwise.
        """
        ...

    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Remove a cache entry asynchronously.

        Parameters:
            id (uuid.UUID): Entry identifier to remove.
        """
        ...

class CacheOptions:
    """Cache options for Hishel 1.0."""

    def __init__(self) -> None:
        """
        Initialize cache options for Hishel 1.0.

        Creates a new CacheOptions instance holding configuration for Hishel's caching behavior; fields are set to their defaults.
        """
        ...

class SyncSqliteStorage(SyncBaseStorage):
    """Synchronous SQLite storage."""

    def __init__(
        self,
        *,
        connection: Any | None = None,
        database_path: str = "hishel_cache.db",
        default_ttl: float | None = None,
        refresh_ttl_on_access: bool = True,
    ) -> None:
        """
        Initialize a synchronous SQLite-backed cache storage.

        Parameters:
            connection (Optional[Any]): Existing DB connection or driver-specific connection object to use instead of opening a new file-based database.
            database_path (str): Path to the SQLite database file used to persist cache entries (default: "hishel_cache.db").
            default_ttl (Optional[float]): Default time-to-live for new cache entries in seconds; if `None`, entries do not receive an implicit TTL.
            refresh_ttl_on_access (bool): If `True`, update an entry's TTL when it is accessed so its lifetime extends; if `False`, accesses do not modify TTL.
        """
        ...
    def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create a cache entry for the given request and response under the specified cache key.

        Parameters:
            request (Request): The request to cache.
            response (Response): The response to cache.
            key (str): Cache key under which the entry will be stored.
            id_ (Optional[uuid.UUID]): Optional explicit UUID to assign to the created entry; a new UUID is generated if omitted.

        Returns:
            Entry: The created cache entry.
        """
        ...
    def get_entries(self, key: str) -> list[Entry]:
        """
        Retrieve all cache entries associated with the given cache key.

        Parameters:
            key (str): Cache lookup key identifying stored entries.

        Returns:
            List[Entry]: List of `Entry` objects matching the key; empty list if no entries are found.
        """
        ...
    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Entry | Callable[[Entry], Entry],
    ) -> Entry | None:
        """
        Update an existing cache entry identified by its UUID.

        Parameters:
            id (uuid.UUID): UUID of the entry to update.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): Either the replacement Entry or a callable that receives the current Entry and returns the updated Entry.

        Returns:
            Optional[Entry]: The updated Entry if the entry existed and was updated, `None` otherwise.
        """
        ...
    def remove_entry(self, id: uuid.UUID) -> None:
        """
        Remove the cache entry with the given identifier from storage.

        Parameters:
            id (uuid.UUID): UUID of the entry to remove.
        """
        ...

class AsyncSqliteStorage(AsyncBaseStorage):
    """Asynchronous SQLite storage."""

    def __init__(
        self,
        *,
        connection: Any | None = None,
        database_path: str = "hishel_cache.db",
        default_ttl: float | None = None,
        refresh_ttl_on_access: bool = True,
    ) -> None:
        """
        Initialize an asynchronous SQLite-backed cache storage for use with async event loops.

        Parameters:
            connection (Optional[Any]): Existing DB connection or driver-specific connection object to use instead of opening a new file-based database.
            database_path (str): Path to the SQLite database file used to persist cache entries (default: "hishel_cache.db").
            default_ttl (Optional[float]): Default time-to-live for new cache entries in seconds; if `None`, entries do not receive an implicit TTL.
            refresh_ttl_on_access (bool): If `True`, update an entry's TTL when it is accessed so its lifetime extends; if `False`, accesses do not modify TTL.
        """
        ...
    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create a cache entry for the given request/response pair under the specified key.

        Parameters:
            request (Request): The original request to be associated with the entry.
            response (Response): The response to be stored in the entry.
            key (str): The cache key under which the entry will be stored.
            id_ (Optional[uuid.UUID]): Optional UUID to assign as the entry's identifier.

        Returns:
            Entry: The created cache entry.
        """
        ...
    async def get_entries(self, key: str) -> list[Entry]:
        """
        Retrieve all cache entries associated with the given cache key.

        Parameters:
            key (str): Cache lookup key identifying stored entries.

        Returns:
            entries (List[Entry]): List of cache Entry objects associated with `key`.
        """
        ...
    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Entry | Callable[[Entry], Entry],
    ) -> Entry | None:
        """
        Update an existing cache entry identified by `id` with new content.

        Parameters:
            id (uuid.UUID): Identifier of the entry to update.
            new_entry (Union[Entry, Callable[[Entry], Entry]]): Either an `Entry` to replace the existing one, or a callable that receives the current `Entry` and returns the updated `Entry`.

        Returns:
            updated_entry (Optional[Entry]): The updated `Entry` if an entry with `id` existed and was updated, `None` otherwise.
        """
        ...
    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Remove the cache entry identified by `id`.

        Parameters:
            id (uuid.UUID): Identifier of the entry to remove.
        """
        ...

class CachePolicy:
    """Cache policy for Hishel 1.0."""

    ...

class SyncCacheProxy:
    """Synchronous cache proxy for requests library."""

    def __init__(
        self,
        request_sender: Any,
        storage: SyncBaseStorage | None = None,
        policy: CachePolicy | None = None,
    ) -> None:
        """
        Create a synchronous cache proxy that wraps a request sender with optional storage and cache policy.

        Parameters:
            request_sender (Any): The callable or object used to perform HTTP requests.
            storage (Optional[SyncBaseStorage]): Optional storage backend used to persist cache entries.
            policy (Optional[CachePolicy]): Optional cache policy that controls caching rules and eviction.
        """
        ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Handle a request using the cache proxy and return the sender's response.

        Arguments passed to this callable are forwarded to the underlying request sender and used to perform the request; the cache proxy may consult storage and policy before delegating.

        Parameters:
            *args: Positional arguments forwarded to the underlying request sender.
            **kwargs: Keyword arguments forwarded to the underlying request sender.

        Returns:
            The response object produced by the underlying request sender.
        """
        ...

class AsyncCacheProxy:
    """Asynchronous cache proxy for aiohttp/httpx."""

    def __init__(
        self,
        client: Any,
        storage: AsyncBaseStorage,
        options: CacheOptions | None = None,
    ) -> None:
        """
        Initialize an asynchronous cache proxy for an HTTP client.

        Parameters:
            client (Any): The HTTP client instance used to send requests (e.g., aiohttp or httpx client).
            storage (AsyncBaseStorage): Asynchronous storage backend used to persist and retrieve cache entries.
            options (Optional[CacheOptions]): Optional cache configuration and behavior overrides.
        """
        ...

    async def handle_request(self, request: Request) -> Response:
        """
        Handle an HTTP request with caching logic.

        Parameters:
            request (Request): The HTTP request to process.

        Returns:
            Response: The HTTP response, either from cache or from the underlying client.
        """
        ...
