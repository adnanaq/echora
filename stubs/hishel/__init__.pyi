"""Type stubs for hishel HTTP caching library (v1.0)."""

import uuid
from typing import Any, Callable, List, Optional, Union

from hishel._core._storages._async_base import AsyncBaseStorage as AsyncBaseStorage

# Re-export storage from _core._storages
from hishel._core._storages._sync_base import SyncBaseStorage as SyncBaseStorage

# Re-export core models from _core.models
from hishel._core.models import Entry as Entry
from hishel._core.models import EntryMeta as EntryMeta
from hishel._core.models import Request as Request
from hishel._core.models import Response as Response

class CacheOptions:
    """Cache options for Hishel 1.0."""

    def __init__(self) -> None: ...

class SyncSqliteStorage(SyncBaseStorage):
    """Synchronous SQLite storage."""

    def __init__(
        self,
        *,
        connection: Optional[Any] = None,
        database_path: str = "hishel_cache.db",
        default_ttl: Optional[float] = None,
        refresh_ttl_on_access: bool = True,
    ) -> None: ...
    def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry: ...
    def get_entries(self, key: str) -> List[Entry]: ...
    def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]: ...
    def remove_entry(self, id: uuid.UUID) -> None: ...

class AsyncSqliteStorage(AsyncBaseStorage):
    """Asynchronous SQLite storage."""

    def __init__(
        self,
        *,
        connection: Optional[Any] = None,
        database_path: str = "hishel_cache.db",
        default_ttl: Optional[float] = None,
        refresh_ttl_on_access: bool = True,
    ) -> None: ...
    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id_: Optional[uuid.UUID] = None,
    ) -> Entry: ...
    async def get_entries(self, key: str) -> List[Entry]: ...
    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]: ...
    async def remove_entry(self, id: uuid.UUID) -> None: ...

class CachePolicy:
    """Cache policy for Hishel 1.0."""

    ...

class SyncCacheProxy:
    """Synchronous cache proxy for requests library."""

    def __init__(
        self,
        request_sender: Any,
        storage: Optional[SyncBaseStorage] = None,
        policy: Optional[CachePolicy] = None,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class AsyncCacheProxy:
    """Asynchronous cache proxy for aiohttp/httpx."""

    def __init__(
        self,
        client: Any,
        storage: AsyncBaseStorage,
        options: Optional[CacheOptions] = None,
    ) -> None: ...
