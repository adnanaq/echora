"""Type stubs for hishel HTTP caching library (v1.0)."""

from typing import Any, Optional

import requests

class CacheOptions:
    """Cache options for Hishel 1.0."""
    def __init__(self) -> None: ...

class SyncBaseStorage:
    """Base synchronous storage."""
    ...

class SyncSqliteStorage(SyncBaseStorage):
    """Synchronous SQLite storage."""
    def __init__(self, sqlite_connection: Optional[Any] = None) -> None: ...

class AsyncBaseStorage:
    """Base asynchronous storage."""
    ...

class AsyncSqliteStorage(AsyncBaseStorage):
    """Asynchronous SQLite storage."""
    def __init__(self, sqlite_connection: Optional[Any] = None) -> None: ...

class SyncCacheProxy:
    """Synchronous cache proxy for requests library."""
    def __init__(
        self,
        client: requests.Session,
        storage: SyncBaseStorage,
        options: Optional[CacheOptions] = None,
    ) -> None: ...

    def get(self, url: str, **kwargs: Any) -> requests.Response: ...
    def post(self, url: str, **kwargs: Any) -> requests.Response: ...
    def put(self, url: str, **kwargs: Any) -> requests.Response: ...
    def delete(self, url: str, **kwargs: Any) -> requests.Response: ...
    def head(self, url: str, **kwargs: Any) -> requests.Response: ...
    def options(self, url: str, **kwargs: Any) -> requests.Response: ...
    def patch(self, url: str, **kwargs: Any) -> requests.Response: ...

class AsyncCacheProxy:
    """Asynchronous cache proxy for aiohttp/httpx."""
    def __init__(
        self,
        client: Any,
        storage: AsyncBaseStorage,
        options: Optional[CacheOptions] = None,
    ) -> None: ...
